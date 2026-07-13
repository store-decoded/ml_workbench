import yaml
from typing import Any
from pydantic import BaseModel, ConfigDict, computed_field


class BBPG(BaseModel):
    host: str = "127.0.0.1"
    port: int = 5432
    username: str = ""
    password: str = ""
    database: str = "bimebazar"
    @computed_field
    @property
    def url(self) -> str:
        return f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
class DataPG(BaseModel):
    host: str = "127.0.0.1"
    port: int = 5432
    username: str = ""
    password: str = ""
    database: str = "postgres"
    @computed_field
    @property
    def url(self) -> str:
        return f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
class Broker(BaseModel):
    host: str = "127.0.0.1"
    port: int = 5432
    topics: list = []
    group_id: str = ""
    @computed_field
    @property
    def url(self) -> str:
        return f"{self.host}:{self.port}"
    
class MinioStorage(BaseModel):
    host: str = '192.168.88.234'
    port: int = 9000
    username: str = ''
    password: str = ''
    secure: bool = False
    @computed_field
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
class Server(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7474
    num_of_workers: int = 2
    api_version: str = "v1"
    log_level: str = "Debug"




class Configs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    DataPG: DataPG
    BBPG: BBPG
    Broker:Broker
    MinioStorage:MinioStorage
    Server:Server
    Parser: Any | None = None
    def __init__(self, config_file_path:str) -> None:
        self.load_config(config_file_path)

    def load_config(self,config_file_path:str):
        with open(config_file_path) as yaml_in:
            yaml_object = yaml.safe_load(yaml_in)
        super().__init__(**yaml_object)

