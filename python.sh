# Install pyenv and dependencies
sudo apt update && sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl libncursesw5-dev \
xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git

# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to your shell config
echo -e '\n# pyenv config' >> ~/.bashrc
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

# Install Python 3.10.13
pyenv install 3.10.13

# Set Python 3.10.13 as global version
pyenv global 3.10.13

# Verify
python --version
