var Confetti = require('./src/confetti.js');
var confetti = new Confetti(300);

// This gets called when someone selects something in the menu during the 
// exit survey... collects data from drop-down menus and submits using mmturkey
function dropdownTip(event){
  var game = event.data.game;
  var data = $(this).find('option:selected').val();
  console.log(data);
  var commands = data.split('::');
  switch(commands[0]) {
  case 'language' :
    game.data = _.extend(game.data, {'nativeEnglish' : commands[1]}); break;
  case 'partner' :
    game.data = _.extend(game.data, {'ratePartner' : commands[1]}); break;
  case 'human' :
    $('#humanResult').show();
    game.data = _.extend(game.data, {'isHuman' : commands[1]}); break;
  case 'didCorrectly' :
    game.data = _.extend(game.data, {'confused' : commands[1]}); break;
  }
}

function submit (event) {
  $('#button_error').show();
  var game = event.data.game;
  game.data = _.extend(game.data, {
    'comments' : $('#comments').val().trim(),
    'strategy' : $('#strategy').val().trim(),
    'partnerStrategy' : $('#partner_strategy').val().trim(),    
    'role' : game.my_role,
    'totalLength' : Date.now() - game.startTime
  });
  game.submitted = true;
  console.log("data is...");
  console.log(game.data);
  console.log(game.socket);
  game.socket.send("exitSurvey." + JSON.stringify(game.data));
  if(_.size(game.urlParams) >= 4) {
    window.opener.turk.submit(game.data, true);
    window.close(); 
  } else {
    console.log("would have submitted the following :")
    console.log(game.data);
  }
}

function disableChatbox(game) {
  $('#chatbox').val('');
  $('#chatbox').attr("disabled", "disabled");
  $('#chatbutton').attr("disabled", "disabled");
}

function highlightFeedback(game, clickedId) {
  // disable chatbox
  disableChatbox(game);
  
  // show highlights as outlines
  var targetcolor = game.my_role == game.playerRoleNames.role1 ? '#5DADE2' : '#FFFFFF';
  var clickedcolor = clickedId == 'target' ? '#32CD32' :'#FF4136';
  $('#target').css({outline: 'solid 10px ' + targetcolor, 'z-index': 2});
  $('#' + clickedId).css({outline: 'solid 10px ' + clickedcolor, 'z-index': 3});  
}

function setupListenerHandlers(game) {
  $('div.pressable').click(function(event) {
    // Only let listener click once they've heard answer back
    if(game.messageSent & !game.alreadyClicked) {
      var clickedId = $(this).attr('id');
      highlightFeedback(game, clickedId);
      game.alreadyClicked = true;
      game.sendResponse(clickedId);
    }
  });
}

function initGrid(game) {
  // Add objects to grid
  _.forEach(game.currStim, (stim, i) => {
    var bkg = 'url(./src/local_imgs/' + stim['url'] + ')';
    var div = $('<div/>')
	  .addClass('pressable')
	  .attr({'id' : stim.targetStatus})
	  .css({'background' : bkg})
	  .css({
	    'position': 'relative',
	    'grid-row': 1, 'grid-column': i+1,
	    'background-size' :'cover'
	  });
    $("#object_grid").append(div);
  });

  // Allow listener to click on things
  game.selections = [];
  console.log(game.my_role);
  if(game.my_role === game.playerRoleNames.role1) {
    console.log('highlighting target');
    $('#target').css({'outline' : 'solid 10px #5DADE2', 'z-index': 2});
  } else if (game.my_role === game.playerRoleNames.role2) {
    console.log('seting up');
    setupListenerHandlers(game);
  }

}

function drawScreen (game) {
  var player = game.getPlayer(game.my_id);
  $('#waiting').html('');
  confetti.reset();
  initGrid(game);    
};

function reset (game, data) {
  game.messageSent = false;
  game.alreadyClicked = false;
  $('#scoreupdate').html(" ");
  if(game.roundNum + 1 > game.numRounds) {
    $('#roundnumber').empty();
    $('#instructs').empty()
      .append("Round\n" + (game.roundNum + 1) + "/" + game.numRounds);
  } else {
    $('#feedback').empty();
    $('#roundnumber').empty()
      .append("Round\n" + (game.roundNum + 1) + "/" + game.numRounds);
  }

  $('#main').show();

  // Clear chatbox for new round
  $('#chatbox').removeAttr("disabled");
  $("#chatbox" ).focus();
  $('#chatbutton').removeAttr("disabled");
  
  // reset labels
  // Update w/ role (can only move stuff if agent)
  $('#roleLabel').empty().append("You are the " + game.my_role + '.');
  $('#instructs').empty();
  if(game.my_role === game.playerRoleNames.role1) {
    $('#instructs')
      .append("<p>Type a message so your partner</p> " +
	      "<p>will click the highlighted image!</p>");
  } else if(game.my_role === game.playerRoleNames.role2) {
    $('#chatarea').hide();
    $('#instructs')
      .append("<p>After your partner types their message, </p>" 
	      + "<p>click the image you think they are telling you about!</p>");
  }
  $("#object_grid").empty();
  drawScreen(game);
}

module.exports = {
  highlightFeedback,
  disableChatbox,
  dropdownTip,
  submit,
  confetti,
  drawScreen,
  reset
};
