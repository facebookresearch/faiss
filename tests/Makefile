MAKEFILE_INC=../makefile.inc

-include $(MAKEFILE_INC)

CC_TESTS = test_ivfpq_indexing test_ivfpq_codec

all: $(CC_TESTS) ../py

run: all
	./run_tests.sh

%: %.cpp  ../libfaiss.a gtest/make/gtest_main.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(BLASLDFLAGS) \
	-Igtest/include -I../..

../libfaiss.a:
	cd ../ && make libfaiss.a

../py:
	cd ../ && make py

gtest/make/gtest_main.a:
	cd gtest/make && make CXX=$(CC) gtest_main.a

clean:
	rm -f $(CC_TESTS)
