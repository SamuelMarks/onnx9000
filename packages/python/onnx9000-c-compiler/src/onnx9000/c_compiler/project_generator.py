"""Project generators for Makefile and CLI wrapper."""


def generate_makefile(prefix: str) -> str:
    """Generate a standalone Makefile for testing."""
    return f"""CC ?= gcc
CFLAGS = -Wall -Wextra -pedantic -std=c89 -O3
TARGET = {prefix}cli
SRCS = {prefix}model.c main.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
\t$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
\t$(CC) $(CFLAGS) -c $< -o $@

clean:
\trm -f $(OBJS) $(TARGET)
"""


def generate_main_c(prefix: str) -> str:
    """Generate a standalone main.c wrapper."""
    return f"""/* Standalone CLI testing wrapper */
#include <stdio.h>
#include <stdlib.h>
#include "{prefix}model.h"

int main(int argc, char** argv) {{
    {prefix}Context ctx;
    uint8_t arena[1024]; /* Dummy arena size for now */
    float input[1]; /* Dummy input */
    float output[1]; /* Dummy output */
    
    (void)argc;
    (void)argv;
    
    if ({prefix}init(&ctx, arena) != 0) {{
        printf("Failed to initialize model.\\n");
        return 1;
    }}
    
    if ({prefix}predict(&ctx, input, output) != 0) {{
        printf("Failed to run prediction.\\n");
        return 1;
    }}
    
    printf("Prediction ran successfully.\\n");
    return 0;
}}
"""


def generate_cmakelists(prefix: str) -> str:
    """Generate a CMakeLists.txt for ESP-IDF integration."""
    return f"""cmake_minimum_required(VERSION 3.16)
project({prefix}model)

add_library({prefix}model STATIC {prefix}model.c)
target_include_directories({prefix}model PUBLIC .)
target_compile_options({prefix}model PRIVATE -Wall -Wextra -std=c99 -O3)
"""


def generate_arduino_sketch(prefix: str) -> str:
    """Generate a .ino Arduino Sketch file."""
    return f"""#include "{prefix}model.h"

{prefix}Context ctx;
uint8_t arena[1024]; // Must be resized manually
float input[1];
float output[1];

void setup() {{
    Serial.begin(115200);
    if ({prefix}init(&ctx, arena) != 0) {{
        Serial.println("Init failed");
    }}
}}

void loop() {{
    {prefix}predict(&ctx, input, output);
    Serial.println(output[0]);
    delay(1000);
}}
"""
