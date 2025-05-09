PWD = $(shell pwd)

# 資料夾名稱
OBJ_DIR := obj

# 所有的 .c 和 .cpp 源文件
SRCS := Thesis.cpp OpenCL.cpp ReadCifar10.cpp ReadMnist.cpp ConvNN.cpp MiniVGG.cpp Layer.cpp common/err_code.cpp

# 所有的目標文件
OBJS := $(addprefix $(OBJ_DIR)/, $(patsubst %.c,%.o,$(filter %.c,$(SRCS))) $(patsubst %.cpp,%.o,$(filter %.cpp,$(SRCS))))

# 目標文件名稱
TARGET := my_program

# 編譯器和編譯選項
CC := gcc
CXX := g++
CFLAGS := -Wall -Werror
CXXFLAGS := -std=c++14 -I$(PWD)/common # Upgrade from 11 to 14

# OpenCV 和 OpenCL 的相關庫
OPENCV_LIBS := $(shell pkg-config --libs opencv4)
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)

OPENCL_LIBS := -framework OpenCL

# 默認目標
all: $(TARGET)

# 編譯規則
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(OPENCV_LIBS) $(OPENCL_LIBS)

$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

# 創造目錄
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
	mkdir -p $(OBJ_DIR)/common

# 清理規則
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean

# export MACOSX_DEPLOYMENT_TARGET=11.7 to eliminate warings