$(document).ready(function () {
    f = function() {
        $("iframe").width('100%');
        $("iframe").height($("iframe").width() * 3 / 4);
    };
    f();
    $(window).resize(function() {
        f();
    });
});