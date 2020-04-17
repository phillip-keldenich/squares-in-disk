_text SEGMENT

PUBLIC ?do_sqrt_rd@impl@ivarp@@YAMM@Z
PUBLIC ?do_sqrt_ru@impl@ivarp@@YAMM@Z
PUBLIC ?do_sqrt_rd@impl@ivarp@@YANN@Z
PUBLIC ?do_sqrt_ru@impl@ivarp@@YANN@Z

?do_sqrt_rd@impl@ivarp@@YAMM@Z PROC EXPORT
	sqrtss xmm0, xmm0
	ret
?do_sqrt_rd@impl@ivarp@@YAMM@Z ENDP

?do_sqrt_ru@impl@ivarp@@YAMM@Z PROC EXPORT
	sub rsp,16
	stmxcsr dword ptr[rsp]
	xor dword ptr[rsp], 6000h
	ldmxcsr dword ptr[rsp]
	sqrtss xmm0, xmm0
	xor dword ptr[rsp], 6000h
	ldmxcsr dword ptr[rsp]
	add rsp,16
	ret
?do_sqrt_ru@impl@ivarp@@YAMM@Z ENDP

?do_sqrt_rd@impl@ivarp@@YANN@Z PROC EXPORT
	sqrtsd xmm0, xmm0
	ret
?do_sqrt_rd@impl@ivarp@@YANN@Z ENDP

?do_sqrt_ru@impl@ivarp@@YANN@Z PROC EXPORT
	sub rsp,16
	stmxcsr dword ptr[rsp]
	xor dword ptr[rsp], 6000h
	ldmxcsr dword ptr[rsp]
	sqrtsd xmm0, xmm0
	xor dword ptr[rsp], 6000h
	ldmxcsr dword ptr[rsp]
	add rsp,16
	ret
?do_sqrt_ru@impl@ivarp@@YANN@Z ENDP

_text ENDS
END
