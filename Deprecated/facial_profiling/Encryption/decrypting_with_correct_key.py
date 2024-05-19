from cryptography.fernet import Fernet

# Encrypted data to brute force (replace this with your actual encrypted data)
encrypted_data = b'gAAAAABmBFyKTj-AsjiyczDFP1UfjdDZVHJsdFWkow4pImJleGz5kIrXsNUT6iKIvqzb3280RFT-j43TVMc2fv_jsiEZVD1_JQ=='  # Your encrypted data here

# Actual key used for encryption 
actual_key = b'5AivBcGbOiirL8CW5FYYU5--GIQ7eD3MgCPBmnqiw20='

# Function to decrypt using the actual key
def decrypt_with_actual_key(encrypted_data, actual_key):
    try:
        f = Fernet(actual_key)
        decrypted_data = f.decrypt(encrypted_data)
        print("Decryption successful with actual key")
        print("Decrypted data:", decrypted_data.decode())
        return True
    except Exception as e:
        print("Decryption failed with actual key:", str(e))
        return False

# Call the decryption function with the actual key
decrypt_with_actual_key(encrypted_data, actual_key)
