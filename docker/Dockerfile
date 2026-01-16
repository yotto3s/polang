# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Arguments for passing Host User ID and Group ID
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    flex \
    bison \
    gcc \
    g++ \
    gdb \
    sudo \
    lsb-release \
    software-properties-common \
    gnupg \
    libzstd-dev \
    libz-dev \
    && apt-get clean

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    sudo ./llvm.sh 20 all && \
    rm llvm.sh

# Install MLIR development packages
RUN apt-get update && apt-get install -y \
    libmlir-20-dev \
    mlir-20-tools \
    && apt-get clean

# Register clang
RUN sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 100 \
    --slave /usr/bin/clang++ clang++ /usr/bin/clang++-20 \
    --slave /usr/bin/clang-format clang-format /usr/bin/clang-format-20 \
    --slave /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-20 \
    --slave /usr/bin/clangd clangd /usr/bin/clangd-20

# Remove default ubuntu user
RUN touch /var/mail/ubuntu && chown ubuntu /var/mail/ubuntu && userdel -r ubuntu

# Create a user with the same UID/GID as the host user and grant sudo privileges
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set the working directory and adjust ownership
WORKDIR /workspace
RUN chown $USER_UID:$USER_GID /workspace

# Switch to the non-root user
USER $USERNAME

# Default command
CMD ["/bin/bash"]
