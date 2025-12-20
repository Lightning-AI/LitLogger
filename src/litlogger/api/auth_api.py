# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from lightning_sdk.lightning_cloud.login import Auth


class AuthApi:
    def __init__(self) -> None:
        self.auth = Auth()

    def authenticate(self) -> bool:
        """Authenticate the user or perform a guest login.

        Returns:
            bool: True if the user is authenticated, False if the user is a guest.
        """
        self.auth.load()

        if getattr(self.auth, "user_id", None) and getattr(self.auth, "api_key", None):
            self.auth.authenticate()
            return True

        self.auth.guest_login()
        return False

    @property
    def guest_id(self) -> str:
        """Get the guest ID.

        Returns:
            str: The guest ID.
        """
        return self.auth.api_key
