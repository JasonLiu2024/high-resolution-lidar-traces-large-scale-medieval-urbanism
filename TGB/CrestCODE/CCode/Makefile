# Makefile for UNIX, linux

CXX        = g++ -O6

all:	IDSet\
	PointTool\
	SvdSolve\
	Eigens\
	Polyhedron\
	setCurvature

Eigens: Eigens.cxx
	$(CXX) -c Eigens.cxx

IDSet: IDSet.cxx
	$(CXX) -c IDSet.cxx
PointTool: PointTool.cxx
	$(CXX) -c PointTool.cxx
SvdSolve: SvdSolve.cxx
	$(CXX) -c SvdSolve.cxx

Polyhedron: Polyhedron.cxx
	$(CXX) -c Polyhedron.cxx
setCurvature: setCurvature.cxx
	$(CXX) -o setCurvature setCurvature.cxx IDSet.o PointTool.o  Eigens.o SvdSolve.o Polyhedron.o -lm
clean:
	-rm -f *.o setCurvature





