/// set/get helpers based on https://www.w3schools.com/js/js_cookies.asp
function setCookie(cname, cvalue, exdays) {
  const d = new Date();
  d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
  document.cookie = cname + "=" + cvalue + ";expires=" + d.toUTCString() + ";SameSite=Strict;path=/";
}

function getCookie(cname) {
  let name = cname + "=";
  let ca = document.cookie.split(';');
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) === ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) === 0) {
      return c.substring(name.length, c.length);
    }
  }
  return "";
}

function modalButtonClose() {
  let modal = document.getElementById('grant-modal');
  modal.style.display = 'none';
  setCookie("grant-modal", true, 365); // might fail if cookies disabled
}

document.addEventListener('DOMContentLoaded', function() {
  let modal = document.getElementById('grant-modal');
  let grantModal = getCookie("grant-modal");
  if (grantModal === false || grantModal === "" || grantModal === null) {
    modal.style.display = 'flex';
  }
});
