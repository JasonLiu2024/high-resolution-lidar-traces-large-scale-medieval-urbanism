# Makefile for UNIX, linux

CXX        = cl -Ox -EHsc

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
	$(CXX) -FesetCurvature.exe setCurvature.cxx IDSet.obj PointTool.obj  Eigens.obj SvdSolve.obj Polyhedron.obj
clean:
	del *.obj setCurvature.exe





