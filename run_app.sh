#!/usr/bin/env sh

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
venv_path="$repository_root/.venv"

if command -v python3 >/dev/null 2>&1; then
    python_command=python3
elif command -v python >/dev/null 2>&1; then
    python_command=python
else
    echo "Python is not installed or is not available on PATH. Install Python 3.14, then run this script again." >&2
    exit 1
fi

cd "$repository_root" || exit 1

if [ ! -d "$venv_path" ]; then
    echo "Creating virtual environment in .venv..."
    "$python_command" -m venv "$venv_path" || exit 1

    printf "Install the optional GPU dependency? [y/N] "
    IFS= read -r gpu_answer || gpu_answer=
    case "$gpu_answer" in
        y|Y|yes|YES|Yes) package='.[gpu]' ;;
        *) package='.' ;;
    esac

    echo "Installing project dependencies..."
    "$venv_path/bin/python" -m pip install -e "$package" || exit 1
fi

activate_script="$venv_path/bin/activate"
if [ ! -f "$activate_script" ]; then
    echo ".venv exists but does not contain a usable Linux/macOS virtual environment. Remove .venv and run this script again." >&2
    exit 1
fi

# shellcheck disable=SC1090
. "$activate_script"
"$venv_path/bin/python" -m gui.test_app
app_exit_code=$?
deactivate
exit "$app_exit_code"
