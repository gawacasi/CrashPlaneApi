import os

# Atualize com suas credenciais
DB_USER = "usuario"
DB_PASSWORD = "senha"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "nome_do_banco"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
