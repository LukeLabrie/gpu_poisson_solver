/* $Id: print.c,v 1.1 2019/12/12 15:03:38 gbarbd Exp gbarbd $ */
#include <stdio.h>
#include <inttypes.h>

static int
is_little_endian(void) {
    int num = 1;
    return (*((char *)&num) == 1);
}

void
print_binary(char *fname, int num, double ***u) {

    FILE *f_ptr;
    size_t written;
    size_t items = num*num*num;

    if ( (f_ptr = fopen(fname, "w")) == NULL ) {
       perror("No output! fopen()");
       return;
    }

    written = fwrite(u[0][0], sizeof(double), items, f_ptr);

    if ( written != items ) {
	    fprintf(stderr, "Writing failed:  only %lu of %lu items saved!\n",
		written, items);
    }

    fclose(f_ptr);
}

void
print_vtk(const char *fname, int n, double ***u) {

    FILE *f_ptr;
    size_t written;
    size_t items = n * n * n;
    size_t i;
    int b;
    unsigned char tmp;

    if ( (f_ptr = fopen(fname, "w")) == NULL ) {
       perror("No output! fopen()");
       return;
    }

    // Write VTK file header
    fprintf(f_ptr, "# vtk DataFile Version 3.0\n");
    fprintf(f_ptr, "saved from function print_vtk.\n");
    fprintf(f_ptr, "BINARY\n");
    fprintf(f_ptr, "DATASET STRUCTURED_POINTS\n");
    fprintf(f_ptr, "DIMENSIONS %d %d %d\n", n, n, n);
    fprintf(f_ptr, "ORIGIN %d %d %d\n", 0, 0, 0);
    fprintf(f_ptr, "SPACING %d %d %d\n", 1, 1, 1);
    fprintf(f_ptr, "POINT_DATA %lu\n", items);
    fprintf(f_ptr, "SCALARS %s %s 1\n", "gray", "double");
    fprintf(f_ptr, "LOOKUP_TABLE default\n");

    if ( is_little_endian() ) {
        // System is little endian, so we need to reverse the byte order.
        written = 0;
        for (i = 0; i < items; ++i) {
            uint64_t crnt = *(uint64_t *)(u[0][0] + i); // Get double as int

            // Reverse byte order and write to file
            crnt = (crnt & 0x00000000FFFFFFFF) << 32 | (crnt & 0xFFFFFFFF00000000) >> 32;
            crnt = (crnt & 0x0000FFFF0000FFFF) << 16 | (crnt & 0xFFFF0000FFFF0000) >> 16;
            crnt = (crnt & 0x00FF00FF00FF00FF) << 8  | (crnt & 0xFF00FF00FF00FF00) >> 8;
            written += fwrite(&crnt, sizeof(uint64_t), 1, f_ptr);
        }
    } else {
        // System is big endian, so just dump the data.
        written = fwrite(u[0][0], sizeof(double), items, f_ptr);
    }

    if ( written != items ) {
	    fprintf(stderr, "Writing failed:  only %lu of %lu items saved!\n",
		written, items);
    }

    fclose(f_ptr);
}
