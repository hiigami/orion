class Credentials(object):
    def __init__(self,
                 username: str,
                 password: str,
                 client_id: str,
                 client_secret: str,
                 app_id: str,
                 version: str) -> None:
        self.app_id = app_id
        self.version = version
        self.username = username
        self.password = password
        self.client_id = client_id
        self.client_secret = client_secret

    def __repr__(self) -> str:
        attributes = [x for x in self.__dict__.keys() if x[:1] != '_']
        values = [getattr(self, x) for x in attributes]
        return "Credentials({0})".format(
            ", ".join(map(lambda p: p, values)))
