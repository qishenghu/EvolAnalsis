import os
import re
import sys
import numpy as np
import torch
import time

from copy import deepcopy
from typing import Any, Callable, Dict, List
from EnvServiceV1.env.env_client import EnvClient
from recipe.beyond_agent.ba_src.beyondagent_execute import context_generate_from_messages_dummy
from recipe.beyond_agent.schema import Experience
from verl.utils.model import compute_position_id_with_mask
from best_logger import print_dict, print_listofdict
from verl.utils.debug.vscode_breakpoint import vscode_conditional_breakpoint
from loguru import logger
from best_logger import register_logger

non_console_mods = ["appworld_io"]
register_logger(non_console_mods=non_console_mods, auto_clean_mods=[], base_log_path="logs/beyondagent", debug=True)


class TempEnvContextManager():
    def __init__(self, env_service_client, env_type, task_id, instance_id):
        self.env_type = env_type
        self.task_id = task_id
        self.env_service_client = env_service_client
        self.instance_id = instance_id

    def __enter__(self):
        init_response = self.env_service_client.create_instance(self.env_type, self.task_id, self.instance_id)
        self.init_response = init_response
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.env_service_client:
            self.env_service_client.release_instance(self.instance_id)
