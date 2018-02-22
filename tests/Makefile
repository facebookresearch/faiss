MAKEFILE_INC=../makefile.inc

-include $(MAKEFILE_INC)

all: test_ivfpq_indexing test_ivfpq_codec

test_ivfpq_indexing: test_ivfpq_indexing.cpp ../libfaiss.a \
	gtest/make/gtest_main.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(BLASLDFLAGS) \
	-Igtest/include -I../..

test_ivfpq_codec: test_ivfpq_codec.cpp ../libfaiss.a \
	gtest/make/gtest_main.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(BLASLDFLAGS) \
	-Igtest/include -I../..

../libfaiss.a:
	cd ../ && make libfaiss.a

gtest/make/gtest_main.a:
	cd gtest/make && make CXX=$(CC) gtest_main.a

clean:
	rm -f test_ivfpq_indexing test_ivfpq_codec
