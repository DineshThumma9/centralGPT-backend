from llama_index.readers.github import GithubRepositoryReader


class CustomRepoReader(GithubRepositoryReader):

    def __init__(self):
        self.__super()