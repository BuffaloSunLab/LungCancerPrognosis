function cost_v = Weight_cost_LS(alpha, P)
d = P.d;
f = P.f;
data = P.data;
v = P.v;
v_new = v-alpha*d;
cost_v = loss_weight(data, f, v_new, P);

end