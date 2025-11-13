# Makefile
CXX=mpicxx
CXXFLAGS=-O3 -std=c++17 -fopenmp
LDFLAGS=
TARGET=qsim
SRCS=main.cpp
OBJS=$(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
