"""The public UI's only route to a prediction.

Before this module the Streamlit app loaded ``model_bundle.pkl`` and called
``predict_proba`` itself, which meant two independent implementations of the
same serving path: the API enforced canonical feature order, validated the
bundle, routed A/B, correlated requests and sanitised errors, and the UI did
none of that. The two could drift silently.

This client owns transport and nothing else. It does not threshold, classify,
map a risk category or choose a variant - every one of those decisions is made
by the API and read back off the response. If a value is not in the response,
the UI does not have it.

``requests`` is used deliberately: it is the HTTP client already pinned in
requirements.lock for the runtime. ``httpx`` is a development-only dependency
here, so reaching for it would have forced a change to a dependency manifest
that another branch owns.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import requests

#: Where the API lives when nothing says otherwise. Loopback, so a developer
#: running `uvicorn app:app` and `streamlit run streamlit_app.py` side by side
#: needs no configuration at all.
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

ENV_BASE_URL = "DIABETES_API_BASE_URL"
ENV_TIMEOUT = "DIABETES_API_TIMEOUT_SECONDS"

#: A bare ``host`` or ``host:port`` with no scheme.
#:
#: Render's Blueprint ``fromService`` with ``property: hostport`` resolves to
#: exactly this shape - "diabetes-api:10000" - because it addresses the service
#: over Render's private network, where there is no TLS terminator and so no
#: scheme to report. Accepting the form lets the Blueprint wire the two
#: services together with a service reference instead of a hardcoded hostname.
_AUTHORITY = re.compile(
    r"^(?P<host>[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)(?::(?P<port>\d{1,5}))?$"
)

#: Generous enough for a cold bundle load on a small instance, short enough that
#: a visitor is told something went wrong rather than left watching a spinner.
DEFAULT_TIMEOUT_SECONDS = 15.0


class ApiError(Exception):
    """A failed call, already reduced to something safe to show a visitor.

    ``user_message`` never contains a traceback, a filesystem path, a database
    detail or raw exception text. ``request_id`` carries the API's correlation
    id when the response supplied one, so an operator can find the server-side
    log for a failure the visitor reports.
    """

    user_message = "Something went wrong while calculating your estimate."

    def __init__(self, message: str = "", request_id: str | None = None) -> None:
        super().__init__(message or self.user_message)
        self.request_id = request_id


class ApiConfigurationError(ApiError):
    """The service address itself is misconfigured.

    Distinct from every other member of this taxonomy: nothing is wrong with
    the API or the network, so retrying cannot help. It is raised when the
    deployment is built, not when a visitor submits, and needs an operator.
    """

    user_message = (
        "This deployment is not configured to reach the risk service, so no "
        "estimate can be calculated. The service operator needs to set the "
        "inference API address."
    )


class ApiUnavailableError(ApiError):
    """The API could not be reached at all - wrong address, or nothing listening."""

    user_message = (
        "The risk service is not reachable right now, so no estimate can be "
        "calculated. Please try again shortly."
    )


class ApiTimeoutError(ApiError):
    """The API accepted the connection but did not answer in time."""

    user_message = (
        "The risk service took too long to respond. Please try again in a moment."
    )


class ApiValidationError(ApiError):
    """The API rejected the submitted values (400/422).

    Reaching this from the UI means the form and the API contract disagree,
    which is a defect rather than something a visitor can correct by retyping.
    """

    user_message = (
        "Some of the submitted answers were not accepted by the risk service. "
        "Please review your answers and try again."
    )


class ModelUnavailableError(ApiError):
    """The API is running but cannot serve predictions (503)."""

    user_message = (
        "The risk model is temporarily unavailable, so no estimate can be "
        "calculated. Please try again later."
    )


class ApiUnexpectedError(ApiError):
    """Any other failure, including an unreadable response body."""

    user_message = (
        "The risk service returned an unexpected response, so no estimate "
        "could be calculated."
    )


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One feature's SHAP contribution, exactly as the API reported it."""

    feature: str
    value: float
    shap_value: float


@dataclass(frozen=True, slots=True)
class Explanation:
    """A SHAP explanation for one scored profile."""

    model_variant: str
    expected_value: float
    contributions: tuple[FeatureContribution, ...]

    def by_feature(self) -> dict[str, float]:
        """SHAP value keyed by feature name, for rendering."""
        return {item.feature: item.shap_value for item in self.contributions}


@dataclass(frozen=True, slots=True)
class Prediction:
    """A scored profile.

    Every field is read from the API response. Nothing here is recomputed by
    the UI - notably ``prediction``, ``risk_category``, ``threshold`` and
    ``model_variant``, which the API decides.
    """

    request_id: str
    model_variant: str
    model_name: str
    prediction: int
    risk_category: str
    probability: float
    threshold: float
    fallback_to_a: bool = False
    confidence_intervals: dict | None = None
    calibration: dict | None = None


def resolve_base_url(explicit: str | None = None) -> str:
    """The API base URL: explicit argument, then environment, then loopback.

    Three accepted forms, in the order an operator is likely to supply them:

    * ``http://host[:port]`` or ``https://host[:port]`` - used as given;
    * ``host[:port]`` with no scheme - the shape Render's ``fromService``
      ``hostport`` produces for private-network addressing, normalised to
      ``http://host[:port]``. Private traffic does not pass a TLS terminator,
      so http is correct rather than a downgrade;
    * nothing at all - loopback, for local development.

    Anything else raises. Silently falling back to loopback on a malformed
    value would turn an operator's typo into a production service that looks
    healthy and can never reach its backend.

    The value is operator-supplied configuration, never visitor input, so this
    is a correctness guard rather than an SSRF boundary - but it still refuses
    schemes other than http/https so a stray ``file://`` or ``ftp://`` cannot
    become a request target.
    """
    candidate = (explicit or os.getenv(ENV_BASE_URL, "") or "").strip()
    if not candidate:
        return DEFAULT_BASE_URL

    candidate = candidate.rstrip("/")
    if not candidate:
        raise ApiConfigurationError(
            f"{ENV_BASE_URL} is only slashes; expected a URL or host:port."
        )

    lowered = candidate.lower()
    if lowered.startswith(("http://", "https://")):
        remainder = candidate.split("://", 1)[1]
        authority = remainder.split("/", 1)[0]
        if not authority or not _AUTHORITY.match(authority):
            raise ApiConfigurationError(
                f"{ENV_BASE_URL} is not a usable URL: {candidate!r}"
            )
        return candidate

    if "://" in candidate:
        scheme = candidate.split("://", 1)[0]
        raise ApiConfigurationError(
            f"{ENV_BASE_URL} must use http or https, not {scheme!r}."
        )

    if _AUTHORITY.match(candidate):
        # Render's hostport form. Normalising here keeps one variable and one
        # contract rather than a second host/port pair to keep in step.
        return f"http://{candidate}"

    raise ApiConfigurationError(
        f"{ENV_BASE_URL} is neither a URL nor a host:port value: {candidate!r}"
    )


def resolve_timeout(explicit: float | None = None) -> float:
    """Request timeout in seconds. A malformed environment value is ignored."""
    if explicit is not None:
        return float(explicit)
    raw = os.getenv(ENV_TIMEOUT, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        parsed = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return parsed if parsed > 0 else DEFAULT_TIMEOUT_SECONDS


class DiabetesApiClient:
    """Typed access to the inference API.

    Resolution happens at construction, so a caller built per Streamlit rerun
    picks up configuration changes without a process restart.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = resolve_base_url(base_url)
        self.timeout = resolve_timeout(timeout)
        self._session = session or requests.Session()

    # ------------------------------------------------------------ transport

    def _post(self, path: str, *, json_body: Any, params: dict) -> dict:
        """One POST, with every failure mode reduced to an ApiError."""
        url = f"{self.base_url}{path}"
        try:
            response = self._session.post(
                url, json=json_body, params=params, timeout=self.timeout
            )
        except requests.exceptions.Timeout as exc:
            raise ApiTimeoutError() from exc
        except requests.exceptions.ConnectionError as exc:
            # Refused, unresolvable, or reset before a response arrived.
            raise ApiUnavailableError() from exc
        except requests.exceptions.RequestException as exc:
            raise ApiUnexpectedError() from exc

        return self._interpret(response)

    def _interpret(self, response: requests.Response) -> dict:
        """Map a status code to either a payload or a deliberate error.

        The body is parsed only for ``detail`` and ``request_id``. Nothing else
        from an error response reaches the caller, so a future API change cannot
        leak server internals into the UI through this path.
        """
        body: dict = {}
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                body = parsed
        except ValueError:
            body = {}

        request_id = body.get("request_id") if isinstance(body.get("request_id"), str) else None

        if response.status_code == 200:
            if not body:
                raise ApiUnexpectedError(request_id=request_id)
            return body
        if response.status_code in (400, 422):
            raise ApiValidationError(request_id=request_id)
        if response.status_code == 404:
            raise ApiUnexpectedError(request_id=request_id)
        if response.status_code == 503:
            raise ModelUnavailableError(request_id=request_id)
        raise ApiUnexpectedError(request_id=request_id)

    # -------------------------------------------------------------- calls

    def predict(
        self,
        features: Mapping[str, float],
        *,
        user_id: str | None = None,
        model_variant: str = "auto",
    ) -> Prediction:
        """Score one profile.

        ``model_variant="auto"`` leaves A/B assignment to the API, which buckets
        deterministically on ``user_id``. The public UI passes a stable
        per-session identifier and reads the chosen variant back off the
        response.

        ``user_id`` is an experiment assignment key, not an authenticated
        identity, and it is genuinely optional: when it is None the parameter is
        omitted from the request entirely rather than sent as a placeholder, so
        the API can record the request as unassigned instead of attributing it
        to a shared fictitious subject.
        """
        params: dict[str, str] = {"model_variant": model_variant}
        if user_id:
            params["user_id"] = user_id

        body = self._post(
            "/predict",
            json_body=dict(features),
            params=params,
        )
        try:
            return Prediction(
                request_id=str(body["request_id"]),
                model_variant=str(body["model_variant"]),
                model_name=str(body["model_name"]),
                prediction=int(body["prediction"]),
                risk_category=str(body["risk_category"]),
                probability=float(body["probability"]),
                threshold=float(body["threshold"]),
                fallback_to_a=bool(body.get("fallback_to_A", False)),
                confidence_intervals=body.get("confidence_intervals"),
                calibration=body.get("calibration"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiUnexpectedError(request_id=body.get("request_id")) from exc

    def explain(
        self,
        features: Mapping[str, float],
        *,
        model_variant: str,
    ) -> Explanation:
        """Per-feature SHAP contributions for the variant that did the scoring.

        The variant is passed explicitly rather than defaulted, so an
        explanation can never describe a different model from the one that
        produced the estimate.
        """
        body = self._post(
            "/explain",
            json_body=dict(features),
            params={"model_variant": model_variant},
        )
        try:
            contributions = tuple(
                FeatureContribution(
                    feature=str(item["feature"]),
                    value=float(item["value"]),
                    shap_value=float(item["shap_value"]),
                )
                for item in body["feature_contributions"]
            )
            return Explanation(
                model_variant=str(body["model_variant"]),
                expected_value=float(body["expected_value"]),
                contributions=contributions,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiUnexpectedError() from exc
