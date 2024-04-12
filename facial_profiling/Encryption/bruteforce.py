from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import base64

# Encrypted data to brute force (replace this with your actual encrypted data)
encrypted_data = b'gAAAAABmBFyKTj-AsjiyczDFP1UfjdDZVHJsdFWkow4pImJleGz5kIrXsNUT6iKIvqzb3280RFT-j43TVMc2fv_jsiEZVD1_JQ=='  

# Brute force function
def brute_force_decrypt(encrypted_data):
    # Generate keys and attempt decryption
    for i in range(10000000):  # Replace 10000000 with your desired number of iterations
        # Convert the integer iteration number to bytes
        iteration_bytes = str(i).encode()
        
        # Derive a key from the iteration number using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # AES-128 requires a 128-bit key (32 bytes)
            salt=b'',  
            iterations=100000,  
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(iteration_bytes))
        
        # Attempt decryption
        try:
            f = Fernet(key)
            decrypted_data = f.decrypt(encrypted_data)
            print(f"Decryption successful with key: {key.decode()}")
            print("Decrypted data:", decrypted_data.decode())
            return True  # Return True if decryption is successful
        except Exception as e:
            print(f"Failed attempt with key: {key.decode()}, Error: {str(e)}")
            continue  # If decryption fails, continue to the next iteration
    print("Brute force failed")
    return False  # Return False if decryption is unsuccessful after all iterations

# Call the brute force function
brute_force_decrypt(encrypted_data)
