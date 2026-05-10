"""Import all module files so their @register_module decorators run on package import."""

from selene.scenarios.modules import step_change, thermal_leak  # noqa: F401
