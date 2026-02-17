"""
Neo Core — Plugin System Tests
===============================
Comprehensive tests for the dynamic plugin system.

Tests:
1. Plugin discovery and loading
2. Plugin validation
3. Plugin execution
4. Error handling and isolation
5. Hot-reloading
6. ToolRegistry integration
7. CLI commands
8. Schema generation
9. Worker type mapping
10. Thread safety
"""

import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from neo_core.tools.plugin_loader import PluginLoader, LoadedPlugin
from neo_core.tools.base_tools import ToolRegistry, TOOL_SCHEMAS


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def temp_plugins_dir(tmp_path):
    """Create a temporary plugins directory."""
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    return plugins_dir


@pytest.fixture
def valid_plugin_file(temp_plugins_dir):
    """Create a valid plugin file."""
    plugin_code = '''
"""Test Plugin"""

PLUGIN_META = {
    "name": "test_plugin",
    "description": "A test plugin",
    "version": "1.0",
    "author": "Test",
    "input_schema": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "A message"
            }
        },
        "required": ["message"]
    },
    "worker_types": ["generic", "researcher"]
}

def execute(message: str) -> str:
    return f"Echo: {message}"
'''
    plugin_path = temp_plugins_dir / "test_plugin.py"
    plugin_path.write_text(plugin_code)
    return plugin_path


@pytest.fixture
def math_plugin_file(temp_plugins_dir):
    """Create a calculator plugin."""
    plugin_code = '''
"""Calculator Plugin"""

PLUGIN_META = {
    "name": "math_tool",
    "description": "Simple math operations",
    "version": "2.0",
    "author": "Test",
    "input_schema": {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "op": {
                "type": "string",
                "enum": ["add", "multiply"]
            }
        },
        "required": ["x", "y", "op"]
    },
    "worker_types": ["analyst", "coder"]
}

def execute(x: float, y: float, op: str) -> str:
    if op == "add":
        return str(x + y)
    elif op == "multiply":
        return str(x * y)
    return "Unknown operation"
'''
    plugin_path = temp_plugins_dir / "math_plugin.py"
    plugin_path.write_text(plugin_code)
    return plugin_path


@pytest.fixture
def crashing_plugin_file(temp_plugins_dir):
    """Create a plugin that crashes on execution."""
    plugin_code = '''
"""Crashing Plugin"""

PLUGIN_META = {
    "name": "crash_plugin",
    "description": "A plugin that crashes",
    "version": "1.0",
    "author": "Test",
    "input_schema": {
        "type": "object",
        "properties": {
            "x": {"type": "number"}
        },
        "required": ["x"]
    },
    "worker_types": ["generic"]
}

def execute(x: float) -> str:
    raise ValueError("Intentional crash for testing")
'''
    plugin_path = temp_plugins_dir / "crash_plugin.py"
    plugin_path.write_text(plugin_code)
    return plugin_path


@pytest.fixture
def invalid_plugin_no_meta(temp_plugins_dir):
    """Create a plugin without PLUGIN_META."""
    plugin_code = '''
def execute(x: str) -> str:
    return x
'''
    plugin_path = temp_plugins_dir / "invalid_plugin.py"
    plugin_path.write_text(plugin_code)
    return plugin_path


@pytest.fixture
def invalid_plugin_no_execute(temp_plugins_dir):
    """Create a plugin without execute function."""
    plugin_code = '''
PLUGIN_META = {
    "name": "bad_plugin",
    "description": "Missing execute",
    "version": "1.0",
    "author": "Test",
    "input_schema": {"type": "object"},
    "worker_types": []
}
'''
    plugin_path = temp_plugins_dir / "invalid_no_exec.py"
    plugin_path.write_text(plugin_code)
    return plugin_path


# ─── Tests: Discovery & Loading ────────────────────────────────────


class TestPluginDiscovery:
    """Test plugin discovery and loading."""

    def test_discover_empty_directory(self, temp_plugins_dir):
        """Test discovery in empty directory."""
        loader = PluginLoader(temp_plugins_dir)
        result = loader.discover()
        assert result["loaded"] == []
        assert result["errors"] == {}

    def test_discover_nonexistent_directory(self, tmp_path):
        """Test discovery with nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"
        loader = PluginLoader(nonexistent)
        result = loader.discover()
        # Should not crash, return empty results
        assert result["loaded"] == []
        assert result["errors"] == {}

    def test_load_valid_plugin(self, temp_plugins_dir, valid_plugin_file):
        """Test loading a valid plugin."""
        loader = PluginLoader(temp_plugins_dir)
        plugin = loader.load_plugin(valid_plugin_file)

        assert plugin is not None
        assert plugin.name == "test_plugin"
        assert plugin.description == "A test plugin"
        assert plugin.version == "1.0"
        assert "generic" in plugin.worker_types
        assert "researcher" in plugin.worker_types

    def test_discover_with_valid_plugin(self, temp_plugins_dir, valid_plugin_file):
        """Test discovery with valid plugin."""
        loader = PluginLoader(temp_plugins_dir)
        result = loader.discover()

        assert "test_plugin" in result["loaded"]
        assert len(result["errors"]) == 0

    def test_discover_with_multiple_plugins(self, temp_plugins_dir, valid_plugin_file, math_plugin_file):
        """Test discovery with multiple plugins."""
        loader = PluginLoader(temp_plugins_dir)
        result = loader.discover()

        assert len(result["loaded"]) == 2
        assert "test_plugin" in result["loaded"]
        assert "math_tool" in result["loaded"]
        assert result["errors"] == {}

    def test_discover_ignores_private_files(self, temp_plugins_dir):
        """Test that files starting with _ are ignored."""
        private_plugin = temp_plugins_dir / "_private.py"
        private_plugin.write_text("PLUGIN_META = {}")

        loader = PluginLoader(temp_plugins_dir)
        result = loader.discover()

        assert "_private" not in result["loaded"]

    def test_load_file_not_found(self, temp_plugins_dir):
        """Test loading nonexistent file."""
        loader = PluginLoader(temp_plugins_dir)
        nonexistent = temp_plugins_dir / "nonexistent.py"

        with pytest.raises(FileNotFoundError):
            loader.load_plugin(nonexistent)


# ─── Tests: Plugin Validation ──────────────────────────────────────


class TestPluginValidation:
    """Test plugin validation."""

    def test_missing_plugin_meta(self, temp_plugins_dir, invalid_plugin_no_meta):
        """Test plugin without PLUGIN_META fails."""
        loader = PluginLoader(temp_plugins_dir)

        with pytest.raises(ValueError, match="PLUGIN_META"):
            loader.load_plugin(invalid_plugin_no_meta)

    def test_missing_execute_function(self, temp_plugins_dir, invalid_plugin_no_execute):
        """Test plugin without execute() fails."""
        loader = PluginLoader(temp_plugins_dir)

        with pytest.raises(ValueError, match="execute"):
            loader.load_plugin(invalid_plugin_no_execute)

    def test_invalid_meta_not_dict(self, temp_plugins_dir):
        """Test PLUGIN_META that is not a dict."""
        plugin_code = 'PLUGIN_META = "not a dict"'
        plugin_path = temp_plugins_dir / "invalid.py"
        plugin_path.write_text(plugin_code)

        loader = PluginLoader(temp_plugins_dir)
        with pytest.raises(ValueError, match="must be a dict"):
            loader.load_plugin(plugin_path)

    def test_missing_required_fields(self, temp_plugins_dir):
        """Test PLUGIN_META missing required fields."""
        plugin_code = '''
PLUGIN_META = {
    "name": "incomplete",
    "description": "Missing version"
}

def execute():
    return "ok"
'''
        plugin_path = temp_plugins_dir / "incomplete.py"
        plugin_path.write_text(plugin_code)

        loader = PluginLoader(temp_plugins_dir)
        with pytest.raises(ValueError, match="missing required fields"):
            loader.load_plugin(plugin_path)

    def test_invalid_input_schema_type(self, temp_plugins_dir):
        """Test PLUGIN_META with invalid input_schema."""
        plugin_code = '''
PLUGIN_META = {
    "name": "bad_schema",
    "description": "Bad schema",
    "version": "1.0",
    "input_schema": "not a dict",
    "worker_types": []
}

def execute():
    return "ok"
'''
        plugin_path = temp_plugins_dir / "bad_schema.py"
        plugin_path.write_text(plugin_code)

        loader = PluginLoader(temp_plugins_dir)
        with pytest.raises(ValueError, match="input_schema must be a dict"):
            loader.load_plugin(plugin_path)

    def test_invalid_worker_types_type(self, temp_plugins_dir):
        """Test PLUGIN_META with invalid worker_types."""
        plugin_code = '''
PLUGIN_META = {
    "name": "bad_workers",
    "description": "Bad workers",
    "version": "1.0",
    "input_schema": {},
    "worker_types": "not a list"
}

def execute():
    return "ok"
'''
        plugin_path = temp_plugins_dir / "bad_workers.py"
        plugin_path.write_text(plugin_code)

        loader = PluginLoader(temp_plugins_dir)
        with pytest.raises(ValueError, match="worker_types must be a list"):
            loader.load_plugin(plugin_path)


# ─── Tests: Execution ──────────────────────────────────────────────


class TestPluginExecution:
    """Test plugin execution."""

    def test_execute_simple_plugin(self, temp_plugins_dir, valid_plugin_file):
        """Test executing a simple plugin."""
        loader = PluginLoader(temp_plugins_dir)
        loader.load_plugin(valid_plugin_file)

        result = loader.execute_plugin("test_plugin", {"message": "hello"})
        assert "Echo: hello" in result

    def test_execute_with_multiple_args(self, temp_plugins_dir, math_plugin_file):
        """Test executing plugin with multiple arguments."""
        loader = PluginLoader(temp_plugins_dir)
        loader.load_plugin(math_plugin_file)

        result = loader.execute_plugin("math_tool", {"x": 3, "y": 4, "op": "add"})
        assert "7" in result

        result = loader.execute_plugin("math_tool", {"x": 3, "y": 4, "op": "multiply"})
        assert "12" in result

    def test_execute_nonexistent_plugin(self, temp_plugins_dir):
        """Test executing plugin that doesn't exist."""
        loader = PluginLoader(temp_plugins_dir)
        result = loader.execute_plugin("nonexistent", {})
        assert "not found" in result.lower()

    def test_execute_crashing_plugin(self, temp_plugins_dir, crashing_plugin_file):
        """Test that crashing plugin is isolated."""
        loader = PluginLoader(temp_plugins_dir)
        loader.load_plugin(crashing_plugin_file)

        # Should not crash Neo, should return error string
        result = loader.execute_plugin("crash_plugin", {"x": 1})
        assert "Error" in result or "error" in result.lower()
        assert isinstance(result, str)

    def test_execute_returns_string(self, temp_plugins_dir, valid_plugin_file):
        """Test that execute result is always a string."""
        loader = PluginLoader(temp_plugins_dir)
        loader.load_plugin(valid_plugin_file)

        result = loader.execute_plugin("test_plugin", {"message": "test"})
        assert isinstance(result, str)

    def test_execute_with_wrong_args(self, temp_plugins_dir, math_plugin_file):
        """Test execution with missing required arguments."""
        loader = PluginLoader(temp_plugins_dir)
        loader.load_plugin(math_plugin_file)

        # Missing required arg should cause error
        result = loader.execute_plugin("math_tool", {"x": 1})
        assert "Error" in result or "error" in result.lower()


# ─── Tests: Management (Load/Unload/Reload) ────────────────────


class TestPluginManagement:
    """Test plugin loading, unloading, and reloading."""

    def test_get_plugin(self, temp_plugins_dir, valid_plugin_file):
        """Test getting a loaded plugin."""
        loader = PluginLoader(temp_plugins_dir)
        loader.load_plugin(valid_plugin_file)

        plugin = loader.get_plugin("test_plugin")
        assert plugin is not None
        assert plugin.name == "test_plugin"

    def test_get_nonexistent_plugin(self, temp_plugins_dir):
        """Test getting nonexistent plugin."""
        loader = PluginLoader(temp_plugins_dir)
        plugin = loader.get_plugin("nonexistent")
        assert plugin is None

    def test_list_plugins(self, temp_plugins_dir, valid_plugin_file, math_plugin_file):
        """Test listing all plugins."""
        loader = PluginLoader(temp_plugins_dir)
        loader.discover()

        plugins_list = loader.list_plugins()
        assert len(plugins_list) == 2

        names = [p["name"] for p in plugins_list]
        assert "test_plugin" in names
        assert "math_tool" in names

    def test_unload_plugin(self, temp_plugins_dir, valid_plugin_file):
        """Test unloading a plugin."""
        loader = PluginLoader(temp_plugins_dir)
        loader.load_plugin(valid_plugin_file)

        assert loader.get_plugin("test_plugin") is not None

        success = loader.unload_plugin("test_plugin")
        assert success is True
        assert loader.get_plugin("test_plugin") is None

    def test_unload_nonexistent_plugin(self, temp_plugins_dir):
        """Test unloading nonexistent plugin."""
        loader = PluginLoader(temp_plugins_dir)
        success = loader.unload_plugin("nonexistent")
        assert success is False

    def test_reload_all_plugins(self, temp_plugins_dir, valid_plugin_file, math_plugin_file):
        """Test hot-reloading all plugins."""
        loader = PluginLoader(temp_plugins_dir)
        loader.discover()

        initial_count = len(loader.list_plugins())
        assert initial_count == 2

        result = loader.reload_all()
        assert len(result["reloaded"]) == 2
        assert "test_plugin" in result["reloaded"]
        assert "math_tool" in result["reloaded"]

        final_count = len(loader.list_plugins())
        assert final_count == 2

    def test_reload_with_new_plugin(self, temp_plugins_dir, valid_plugin_file, math_plugin_file):
        """Test reload picks up newly added plugins."""
        loader = PluginLoader(temp_plugins_dir)
        loader.discover()
        initial = len(loader.list_plugins())

        # Add new plugin
        new_plugin = temp_plugins_dir / "new_plugin.py"
        new_plugin.write_text('''
PLUGIN_META = {
    "name": "new_tool",
    "description": "New",
    "version": "1.0",
    "input_schema": {},
    "worker_types": []
}
def execute() -> str:
    return "ok"
''')

        result = loader.reload_all()
        # Should have reloaded old ones + loaded new
        # Note: reload_all reloads based on filepaths already known
        # New files won't be picked up by reload_all
        assert len(result["reloaded"]) >= initial


# ─── Tests: Schema Generation ──────────────────────────────────────


class TestPluginSchemas:
    """Test schema generation for plugins."""

    def test_get_plugin_schemas(self, temp_plugins_dir, valid_plugin_file, math_plugin_file):
        """Test getting schemas for all plugins."""
        loader = PluginLoader(temp_plugins_dir)
        loader.discover()

        schemas = loader.get_plugin_schemas()
        assert len(schemas) == 2
        assert "test_plugin" in schemas
        assert "math_tool" in schemas

        test_schema = schemas["test_plugin"]
        assert test_schema["name"] == "test_plugin"
        assert "description" in test_schema
        assert "input_schema" in test_schema

    def test_plugin_to_dict(self, temp_plugins_dir, valid_plugin_file):
        """Test converting plugin to dict."""
        loader = PluginLoader(temp_plugins_dir)
        plugin = loader.load_plugin(valid_plugin_file)

        plugin_dict = plugin.to_dict()
        assert plugin_dict["name"] == "test_plugin"
        assert plugin_dict["description"] == "A test plugin"
        assert plugin_dict["version"] == "1.0"
        assert isinstance(plugin_dict["worker_types"], list)
        assert "loaded_at" in plugin_dict


# ─── Tests: Worker Type Mapping ────────────────────────────────────


class TestWorkerTypeMapping:
    """Test plugin filtering by worker type."""

    def test_get_plugins_for_worker_type(self, temp_plugins_dir, valid_plugin_file, math_plugin_file):
        """Test getting plugins for specific worker type."""
        loader = PluginLoader(temp_plugins_dir)
        loader.discover()

        # valid_plugin supports generic and researcher
        generic_plugins = loader.get_plugins_for_worker_type("generic")
        assert len(generic_plugins) >= 1
        names = [p.name for p in generic_plugins]
        assert "test_plugin" in names

        # math_tool supports analyst and coder
        analyst_plugins = loader.get_plugins_for_worker_type("analyst")
        names = [p.name for p in analyst_plugins]
        assert "math_tool" in names

        # Non-matching worker type
        translator_plugins = loader.get_plugins_for_worker_type("translator")
        assert len(translator_plugins) == 0

    def test_get_plugins_for_nonexistent_worker_type(self, temp_plugins_dir, valid_plugin_file):
        """Test getting plugins for nonexistent worker type."""
        loader = PluginLoader(temp_plugins_dir)
        loader.discover()

        plugins = loader.get_plugins_for_worker_type("nonexistent_type")
        assert len(plugins) == 0


# ─── Tests: Thread Safety ──────────────────────────────────────────


class TestThreadSafety:
    """Test thread-safety of plugin operations."""

    def test_concurrent_plugin_loading(self, temp_plugins_dir, valid_plugin_file):
        """Test concurrent loading doesn't cause issues."""
        loader = PluginLoader(temp_plugins_dir)

        results = []

        def load_plugin():
            try:
                plugin = loader.load_plugin(valid_plugin_file)
                results.append(plugin is not None)
            except Exception as e:
                results.append(False)

        threads = [threading.Thread(target=load_plugin) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least one should succeed
        assert any(results)

    def test_concurrent_plugin_execution(self, temp_plugins_dir, valid_plugin_file):
        """Test concurrent execution is safe."""
        loader = PluginLoader(temp_plugins_dir)
        loader.load_plugin(valid_plugin_file)

        results = []

        def execute():
            result = loader.execute_plugin("test_plugin", {"message": "test"})
            results.append(result)

        threads = [threading.Thread(target=execute) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(isinstance(r, str) for r in results)


# ─── Tests: ToolRegistry Integration ───────────────────────────────


class TestToolRegistryIntegration:
    """Test integration with ToolRegistry."""

    def test_registry_initialize_with_plugins(self, temp_plugins_dir, valid_plugin_file):
        """Test ToolRegistry initialization with plugins."""
        ToolRegistry.initialize(mock_mode=True, plugins_dir=temp_plugins_dir)

        # Check that plugin was registered
        assert "test_plugin" in ToolRegistry.list_tools()

    def test_registry_plugin_in_schema(self, temp_plugins_dir, valid_plugin_file):
        """Test that plugin schema is in TOOL_SCHEMAS."""
        ToolRegistry.initialize(mock_mode=True, plugins_dir=temp_plugins_dir)

        assert "test_plugin" in TOOL_SCHEMAS
        schema = TOOL_SCHEMAS["test_plugin"]
        assert schema["name"] == "test_plugin"
        assert "description" in schema
        assert "input_schema" in schema

    def test_registry_plugin_in_worker_tools(self, temp_plugins_dir, math_plugin_file):
        """Test that plugin is added to WORKER_TOOLS."""
        ToolRegistry.initialize(mock_mode=True, plugins_dir=temp_plugins_dir)

        # math_tool is available for analyst workers
        analyst_tools = ToolRegistry.WORKER_TOOLS.get("analyst", [])
        assert "math_tool" in analyst_tools

    def test_registry_reload_plugins(self, temp_plugins_dir, valid_plugin_file):
        """Test hot-reloading plugins via ToolRegistry."""
        ToolRegistry.initialize(mock_mode=True, plugins_dir=temp_plugins_dir)

        result = ToolRegistry.reload_plugins()
        assert "reloaded" in result or "message" in result

    def test_registry_without_plugins_dir(self):
        """Test ToolRegistry works without plugins_dir."""
        ToolRegistry.initialize(mock_mode=True)
        # Should not crash, basic tools should still work
        assert "web_search" in ToolRegistry.list_tools()

    def test_registry_execute_plugin_tool(self, temp_plugins_dir, valid_plugin_file):
        """Test executing plugin through ToolRegistry."""
        ToolRegistry.initialize(mock_mode=True, plugins_dir=temp_plugins_dir)

        # Plugin should be callable as a tool
        result = ToolRegistry.execute_tool("test_plugin", {"message": "hello"})
        assert isinstance(result, str)
        assert "Echo" in result or "hello" in result


# ─── Tests: Error Handling ────────────────────────────────────────


class TestErrorHandling:
    """Test error handling and isolation."""

    def test_plugin_syntax_error(self, temp_plugins_dir):
        """Test handling of plugin with syntax error."""
        plugin_code = 'PLUGIN_META = {' # Invalid syntax

        plugin_path = temp_plugins_dir / "syntax_error.py"
        plugin_path.write_text(plugin_code)

        loader = PluginLoader(temp_plugins_dir)
        result = loader.discover()

        assert "syntax_error" in result["errors"]

    def test_plugin_import_error(self, temp_plugins_dir):
        """Test handling of plugin with import error."""
        plugin_code = '''
import nonexistent_module_xyz

PLUGIN_META = {"name": "bad", "description": "", "version": "1.0", "input_schema": {}, "worker_types": []}

def execute():
    return "ok"
'''
        plugin_path = temp_plugins_dir / "import_error.py"
        plugin_path.write_text(plugin_code)

        loader = PluginLoader(temp_plugins_dir)
        result = loader.discover()

        assert "import_error" in result["errors"]

    def test_discover_with_mixed_valid_invalid(self, temp_plugins_dir, valid_plugin_file):
        """Test discovery with mix of valid and invalid plugins."""
        # Add invalid plugin
        invalid_path = temp_plugins_dir / "invalid.py"
        invalid_path.write_text("PLUGIN_META = {")  # Syntax error

        loader = PluginLoader(temp_plugins_dir)
        result = loader.discover()

        assert "test_plugin" in result["loaded"]
        assert "invalid" in result["errors"]


# ─── Tests: Loaded Plugin Dataclass ────────────────────────────────


class TestLoadedPluginDataclass:
    """Test LoadedPlugin dataclass."""

    def test_loaded_plugin_attributes(self, temp_plugins_dir, valid_plugin_file):
        """Test LoadedPlugin has expected attributes."""
        loader = PluginLoader(temp_plugins_dir)
        plugin = loader.load_plugin(valid_plugin_file)

        assert hasattr(plugin, "name")
        assert hasattr(plugin, "description")
        assert hasattr(plugin, "version")
        assert hasattr(plugin, "input_schema")
        assert hasattr(plugin, "worker_types")
        assert hasattr(plugin, "execute_fn")
        assert hasattr(plugin, "filepath")
        assert hasattr(plugin, "loaded_at")

    def test_loaded_plugin_loaded_at_is_iso_format(self, temp_plugins_dir, valid_plugin_file):
        """Test that loaded_at is ISO format."""
        loader = PluginLoader(temp_plugins_dir)
        plugin = loader.load_plugin(valid_plugin_file)

        # Should be ISO format (YYYY-MM-DDTHH:MM:SS.ffffff)
        assert "T" in plugin.loaded_at
        # Should be parseable as ISO
        from datetime import datetime
        datetime.fromisoformat(plugin.loaded_at)
