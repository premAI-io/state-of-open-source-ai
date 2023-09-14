/// set/get helpers based on https://www.w3schools.com/js/js_cookies.asp
function setCookie(cname, cvalue, exdays) {
  const d = new Date();
  d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
  document.cookie = cname + "=" + cvalue + ";expires=" + d.toUTCString() + ";SameSite=Strict;path=/";
}

function getCookie(cname) {
  let name = cname + "=";
  let ca = document.cookie.split(';');
  for(let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) == ' ') { c = c.substring(1); }
    if (c.indexOf(name) == 0) {
      return c.substring(name.length, c.length);
    }
  }
  return "";
}

async function checkCookie() {
  let email = getCookie("address");
  let prefix = "";
  while (email == "" || email == null) {
    email = prompt(prefix + "To access this book for free, please enter your email. We won't spam you.", "");
    if (email != "" && email != null) {
      let res = await fetch("https://premai.pythonanywhere.com/email?a=" + email);
      if (200 <= res.status && res.status < 300) {
        setCookie("address", email, 365);
      } else {
        email = "";
        msg = await res.json();
        prefix = "Error " + res.status + ": " + msg.status + "\n\n";
      }
    }
  }
}
