import random
import string


def create_and_save_phone_book(
    n_entries: int, n_digits: int, name_length: int, filename: str
) -> None:
    """
    Creates a random phone book with n_entries entries and saves it to a file.
    Each entry has a name of length name_length and a phone number of n_digits.
    Writes to a file named filename, with each entry on one line.
    """
    phone_book = []
    for i in range(n_entries):
        name = "".join(random.choices(string.ascii_lowercase, k=name_length))
        phone_number = "".join(random.choices(string.digits, k=n_digits))
        phone_number_formatted = (
            (f"({phone_number[:3]}) {phone_number[3:6]}-{phone_number[6:]}")
            if n_digits >= 10
            else phone_number
        )
        phone_book.append(f"{name} : {phone_number_formatted}")
    with open(filename, "w") as f:
        f.write("\n".join(phone_book))


if __name__ == "__main__":
    create_and_save_phone_book(1000000, 10, 15, "phone_book.txt")
