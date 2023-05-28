from typing import Optional
from pydantic import BaseSettings, SecretStr

class TelegramSecrets(BaseSettings):
    API_TOKEN: Optional[SecretStr]