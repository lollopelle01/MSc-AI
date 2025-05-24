FROM minizinc/minizinc:latest

WORKDIR /project

COPY . .

# Install necessary packages
RUN apt-get update \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && apt-get install -y glpk-utils \
    && python3 -m pip install -r requirements.txt --break-system-packages \
    && rm -rf /var/lib/apt/lists/*

# As default prints the help 
RUN echo 'python3 /project/main.py -h' >> /root/.bashrc
