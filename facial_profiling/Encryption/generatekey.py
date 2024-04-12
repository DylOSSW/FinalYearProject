from cryptography.fernet import Fernet

# Generate a Fernet key
fernet_key = Fernet.generate_key()

# Write the Fernet key to a file
with open('config.key', 'wb') as fernet_key_file:
    fernet_key_file.write(fernet_key)
    
key = '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'
with open('1111openai_api.key', 'wb') as key_file:
    key_file.write(key.encode())  # Encode the string to bytes before writing



