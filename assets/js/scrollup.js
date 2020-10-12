$(document).ready(function () {
    var mobile = screen.width < 500 ||
        navigator.userAgent.match(/Android/i) ||
        navigator.userAgent.match(/webOS/i) ||
        navigator.userAgent.match(/iPhone/i) ||
        navigator.userAgent.match(/iPod/i);

    $(window).scroll(function () {
        if (!mobile) {
            if ($(this).scrollTop() > 100) {
                $('.scrollup').show();
                $('.icon-scrollup').fadeIn();
            } else {
                $('.icon-scrollup').fadeOut(function(){
                    $('.scrollup').hide();
                });
            }
        }
    });
});
