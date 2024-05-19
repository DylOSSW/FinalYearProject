def caesar_encrypt(text, shift):
    result = ""

    for i in range(len(text)):
        char = text[i]

        # Encrypt uppercase characters
        if char.isupper():
            result += chr((ord(char) + shift - 65) % 26 + 65)
        # Encrypt lowercase characters
        elif char.islower():
            result += chr((ord(char) + shift - 97) % 26 + 97)
        else:
            result += char

    return result

def caesar_decrypt(ciphertext, shift):
    return caesar_encrypt(ciphertext, -shift)

# Example usage
plaintext = "Dylan"
shift = 4
encrypted = caesar_encrypt(plaintext, shift)
decrypted = caesar_decrypt(encrypted, shift)

print(f"Plaintext: {plaintext}")
print(f"Encrypted: {encrypted}")
print(f"Decrypted: {decrypted}")

def brute_force_caesar(ciphertext):
    for key in range(26):
        potential_plaintext = caesar_decrypt(ciphertext, key)
        print(f"Shift {key}: {potential_plaintext}")

# Example usage with the encrypted text from before
brute_force_caesar(encrypted)
