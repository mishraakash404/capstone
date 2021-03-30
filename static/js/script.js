$(document).ready(function(){
		// $('.logo').css({'transform':'translate(150%,150%)','Position':'absolute',});
		// $('.logo').animate({'background':'red'});
		$('.content .left-side Button').click(function(){
			$('.content .right-side .back').fadeToggle(200);
			$('.content .right-side .work-box').delay(200).fadeToggle(200);
		});
		function show(input){
		if (input.files && input.files[0]) {
			var reader= new FileReader();
			reader.addEventListener('load',function(){
				var value=this.result;
				$('.selected').attr('src',value);
			})
			reader.readAsDataURL(input.files[0]);
		}
		else{
			$('.selected').attr('src','');
		}
	}
	
	$('#img-name').change(function(){
		show(this);
		// var value=e.target.files[0].name;
		// $('img').attr('src',value);
	});
});	