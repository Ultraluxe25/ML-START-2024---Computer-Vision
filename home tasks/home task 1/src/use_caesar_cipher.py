def use_caesar_cipher(shift: int, text: str) -> str:
    """
    Current function uses the Caesar cipher to encrypt a message
    """
    alphabet = " abcdefghijklmnopqrstuvwxyz"
    alphabet_dict = {
        char: index for index, char in enumerate(alphabet)
    }  # Dict comprehension
    n = len(alphabet)
    text = text.strip()  # Eliminates whitespaces
    result = []

    for char in text:
        if char in alphabet_dict:
            index = alphabet_dict[char]  # O(1) Time complexity
            new_index = (index + shift) % n
            result.append(alphabet[new_index])
        else:
            result.append(char)  # Keeps unchanged symbols not in alphabet

    return "".join(result)


if __name__ == "__main__":
    shift = int(input("Please input the shift: "))
    message = input("Please input the message: ")
    print(f"Result is: {use_caesar_cipher(shift, message)}")
