import sys
from networksecurity.logging import logger

class NbaException(Exception):
    def __init__(self, error_message, error_details:sys):
        self.error_message = error_message
        _, _, exc_tv = error_details.exc_info()

        self.lineno = exc_tv.tb_lineno
        self.filename = exc_tv.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error occurred in script: {self.filename} at line number: {self.lineno}. Error message: {self.error_message}"
