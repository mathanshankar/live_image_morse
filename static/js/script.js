const socket = io();

socket.on('update_caption', function (data) {
    document.getElementById('object-label').innerText = data.label;
    document.getElementById('morse-code').innerText = data.morse;
});
