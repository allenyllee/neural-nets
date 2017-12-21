# include <stdio.h>

typedef struct _Unit Unit;
typedef struct _Neural Neural;
typedef Unit* (*forward_func) (Neural *, Unit*, Unit*);
typedef void (*backward_func) (Neural *);

struct _Unit{
    float value;
    float grad;
};

struct _Neural {
    Neural *this;
    Unit *u1;
    Unit *u2;
    Unit uout;
    forward_func forward;
    backward_func backward;
};

Unit* add(Neural * const this, Unit *a, Unit *b){
    this->u1 = a;
    this->u2 = b;
    this->uout.value = this->u1->value + this->u2->value;
    return &this->uout;
}

void addgrad(Neural * const this){
    this->u1->grad += 1.0 * this->uout.grad;
    this->u2->grad += 1.0 * this->uout.grad;
}

void init_add(Neural *const x){
    x->this = x;
    x->forward = add;
    x->backward = addgrad;
}

Unit* mul(Neural * const this, Unit *a, Unit *b){
    this->u1 = a;
    this->u2 = b;
    this->uout.value = this->u1->value * this->u2->value;
    return &this->uout;
}

void mulgrad(Neural * const this){
    this->u1->grad += this->u2->value * this->uout.grad;
    this->u2->grad += this->u1->value * this->uout.grad;
}

void init_mul(Neural *const x){
    x->this = x;
    x->forward = mul;
    x->backward = mulgrad;
}


int main(){
    Unit x = {.value=2,     .grad=0};
    Unit y = {.value=-1.5,  .grad=0};
    Unit z = {.value=-6,    .grad=0};
    Unit *tmp = NULL;
    float step_size = 0.001;

    Neural addgate1, mulgate1;

    init_add(&addgate1);
    init_mul(&mulgate1);

    // evaluate value
    tmp = addgate1.forward(addgate1.this, &x, &y);
    tmp = mulgate1.forward(mulgate1.this, tmp, &z);

    printf("forward value = %f\n", tmp->value);

    // evaluate grad
    mulgate1.uout.grad=1;
    mulgate1.backward(mulgate1.this);
    addgate1.backward(addgate1.this);

    // update input varible
    x.value += step_size * x.grad;
    y.value += step_size * y.grad;
    z.value += step_size * z.grad;

    // evaluate value
    tmp = addgate1.forward(addgate1.this, &x, &y);
    tmp = mulgate1.forward(mulgate1.this, tmp, &z);

    printf("forward value = %f\n", tmp->value);

    return 0;
}