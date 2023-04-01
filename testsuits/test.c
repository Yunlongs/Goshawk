#include <stdlib.h>
#include <stdio.h>

typedef struct node
{
    int val;
    struct node *next;
} Node;

Node *create()
{
    return (Node *)
        malloc(sizeof(Node));
}

void myfree(Node **n)
{
    free(*n);
    //*n = NULL;
}

int main()
{
    Node *n1, *n2;
    n1 = create();
    n1->val = 10;
    n2 = create();
    n2->val = 5;
    int y, i = 0;
    scanf("%d", &y);

    while (y < 10)
    {
        int x;
        scanf("%d", &x);

        while (x > 10)
        {
            i++;
            if (i > 4)
                x += i;
            else
                x -= i;

            if (x % 2 == 0)
                n2 = n1;
        }

        y += x;
    }

    myfree(&n1);
    printf("%d", n1->val); // bug: UAF
    myfree(&n2);
}