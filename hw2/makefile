include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

bvp: bvp.o
	-${CLINKER} -o bvp bvp.o  ${PETSC_LIB}
	${RM} bvp.o

.PHONY: clean distclean

clean::
	@rm -f *~ bvp *tmp
	@rm -f *.dat *.dat.info
distclean: clean