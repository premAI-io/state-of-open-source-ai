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

async function emailButtonClick() {
  let emailInput = document.getElementById("email-input");
  // from https://html.spec.whatwg.org/multipage/input.html#valid-e-mail-address
  const valid = /^[a-zA-Z0-9.!#$%&'*+\/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/;
  if (valid.test(emailInput.value)) {
    let modal = document.getElementById('email-modal');
    modal.style.display = 'none';
    setCookie("email", emailInput.value, 365); // might fail if cookies disabled
  } else {
    let emailError = document.getElementsByClassName('email-error')[0];
    emailError.innerHTML = "Error: please enter a valid email";
  }
}

document.addEventListener('DOMContentLoaded', function() {
  let modal = document.getElementById('email-modal');
  let emailCookie = getCookie("email");
  let emailInput = document.getElementById("email-input");
  if (emailCookie === false || emailCookie === "" || emailCookie === null) {
    modal.style.display = 'flex';
    emailInput.value = "";
  }
  emailInput.focus()
  // When user click Enter, click the submit button
  emailInput.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      document.getElementById("email-submit").click();
    }
  });
});
