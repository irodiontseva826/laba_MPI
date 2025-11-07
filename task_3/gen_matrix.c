#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    if (argc != 3) return 1;
    int N = atoi(argv[1]);
    FILE* f = fopen(argv[2], "w");
    if (!f) return 1;
    srand(time(NULL) + N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(f, "%d", rand() % 5 + 1);
            if (j < N-1) fprintf(f, " ");
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return 0;
}
