from admin_auth import create_or_update_user


def main() -> None:
    print("Create or update admin user")
    username = input("Username: ").strip()
    password = input("Password (min 8 chars): ").strip()
    create_or_update_user(username, password)
    print(f"Admin user '{username}' saved successfully.")


if __name__ == "__main__":
    main()
