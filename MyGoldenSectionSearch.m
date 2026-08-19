function [s_opt v_opt]=MyGoldenSectionSearch(LineFunc,smax,P,v_end)
s_start=0;
s_end=smax;
v_opt=v_end;
s_opt=s_end;

s1=s_start+(s_end-s_start)*0.618;
s2=s_start+(s_end-s_start)*0.382;
v1=LineFunc(s1,P);
v2=LineFunc(s2,P);

for it=1:10
    if (v1 >v_opt && v2 > v_opt)
        return
    end
    if (v1 > v2)
        v_opt=v2;
		s_opt=s2;
		v1=v2;
		s_end=s1;
		s1=s2;
		s2=s_start+(s_end-s_start)*0.382;
        v2=LineFunc(s2,P);
    else
		v_opt=v1;
		s_opt=s1;
		v2=v1;
		s_start=s2;
		s2=s1;
		s1=s_start+(s_end-s_start)*0.618;
		v1=LineFunc(s1,P);
    end
end
if (v1 < v_opt)
	v_opt=v1;
	s_opt=s1;
end
if (v2 < v_opt)
	v_opt=v2;
	s_opt=s2;
end

end