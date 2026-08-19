function cost_label = Label_cost_LS(alpha, P)
d = P.d;
f = P.f;
A_mat = P.A_mat;
lambda = P.lambda;
y_enlarged = P.y;
bag_ids = P.bag_ids;
t = P.t;
f_new = f-alpha*d;
[loss_label_smooth, loss_label_y] = loss_label(A_mat, f_new, y_enlarged, bag_ids, t);
cost_label = lambda*loss_label_smooth + loss_label_y;

end
