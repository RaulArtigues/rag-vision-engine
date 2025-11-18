from starlette_context.errors import ContextDoesNotExistError
from contextlib import asynccontextmanager
from colorama import Fore, Style, init
from fastapi import APIRouter, FastAPI
from starlette_context import context
from datetime import datetime, UTC
from typing import Optional, Union
import logging
import inspect
import os

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(message)s") 
logger = logging.getLogger("app_logger")
router = APIRouter()

class LoggerManager:
    """
    A class that centralizes logging functions, including formatting logs and retrieving
    additional details such as RequestID, script path, and class name.
    """
    @staticmethod
    def get_relative_path(filepath):
        """
        Get the relative path of the file in relation to the project's root directory.

        Args:
            filepath (str): The absolute path of the file.

        Returns:
            str: The relative path of the file.
        """
        project_root = os.getcwd()
        return os.path.relpath(filepath, project_root)

    @staticmethod
    def get_class_name(frame, cls=None):
        """
        Retrieve the class name from the current execution frame.

        Args:
            frame (frame): The execution frame from which the function is invoked.
            cls (type, optional): The class type, if available.

        Returns:
            str: The name of the class if found; otherwise, returns 'N/A'.
        """
        if cls:
            return cls.__name__
        local_vars = frame.f_locals

        if 'self' in local_vars:
            return local_vars['self'].__class__.__name__
        if 'cls' in local_vars:
            return local_vars['cls'].__name__
        return "N/A"

    @staticmethod
    def get_class_name(frame):
        """
        Retrieve the class name from the current execution frame.

        Args:
            frame (frame): The execution frame from which the function is invoked.

        Returns:
            str: The name of the class if found; otherwise, returns 'N/A'.
        """
        local_vars = frame.f_locals
        if 'self' in local_vars:
            return local_vars['self'].__class__.__name__
        return "N/A"

    @staticmethod
    def log_formatter(message: str, imageId: str, code: int, level: str = "INFO"):
        """
        Format and log a message with additional details such as RequestID, script path,
        class name, and function name. Applies colors based on the log level.

        Args:
            message (str): The main message of the log.
            processId (str): Unique identifier of the processing instance that groups related images and tasks.
            imageId (str): Unique identifier of session for proper traceability.
            useCase (str): Origin of the request (e.g., 'vda', 'vdc', etc.).
            code (int): Status code associated with the event.
            http_code (int): HTTP status code associated with the event.
            level (str): Log level (INFO, ERROR, WARNING, DEBUG). Defaults to 'INFO'.

        Returns:
            None
        """
        timestamp = datetime.now(UTC).isoformat()

        caller_frame = inspect.stack()[1]
        relative_path = LoggerManager.get_relative_path(caller_frame.filename)
        class_name = LoggerManager.get_class_name(caller_frame.frame)
        function_name = caller_frame.function

        if level == "INFO":
            level_text = f"{Fore.GREEN}[INFO]{Style.RESET_ALL}"
        elif level == "ERROR":
            level_text = f"{Fore.RED}[ERROR]{Style.RESET_ALL}"
        elif level == "WARNING":
            level_text = f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL}"
        else:
            level_text = f"[{level}]"

        log_message = (
            f"loggingLevel: {level_text} | @timestamp: {timestamp} | " f"imageId: {imageId} | Code: {code} | loggingMessage: {message} | " f"pathScript: {relative_path} | pythonClass: {class_name}| pythonFunction: {function_name}")

        if level == "INFO":
            logger.info(log_message)
        elif level == "ERROR":
            logger.error(log_message)
        elif level == "WARNING":
            logger.warning(log_message)
        else:
            logger.debug(log_message)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI startup event.
    """
    pass

app = FastAPI(lifespan=lifespan)