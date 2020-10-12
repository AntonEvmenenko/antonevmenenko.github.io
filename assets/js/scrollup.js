$(document).ready(function () {
    let mobile = screen.width < 500 ||
        navigator.userAgent.match(/Android/i) ||
        navigator.userAgent.match(/webOS/i) ||
        navigator.userAgent.match(/iPhone/i) ||
        navigator.userAgent.match(/iPod/i);
    let previousPosition = 0;
    let autoScroll = false;
    let rotateScrollupIcon = function (rotate) {
        $('.icon-scrollup').css('transform', 'translate(-50%, -50%) ' + (rotate ? 'rotate(180deg)' : ''));
    };

    $(window).scroll(function () {
        if (autoScroll || mobile) {
            return;
        }

        var position = $(document).scrollTop();

        if (position != 0 && position != previousPosition) {
            previousPosition = 0;
            rotateScrollupIcon(false);
        }

        if ($(this).scrollTop() > 100) {
            $('.scrollup').show();
            $('.icon-scrollup').fadeIn();
        } else {
            if (previousPosition == 0) {
                $('.icon-scrollup').fadeOut(function () {
                    $('.scrollup').hide();
                });
            }
        }
    });

    $('.scrollup').click(function () {
        if (mobile) {
            return;
        }

        let currentPosition = $(document).scrollTop();
        let goalPosition = (currentPosition == 0 && previousPosition != 0) ? previousPosition : 0;

        previousPosition = currentPosition;
        rotateScrollupIcon(goalPosition == 0);
        autoScroll = true;
        $("html, body").animate({
            scrollTop: goalPosition
        }, 600, function () {
            autoScroll = false;
        });
        return false;
    });
});