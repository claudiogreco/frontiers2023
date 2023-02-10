var UI = require('./drawing.js');

// Update client versions of variables with data received from
// server_send_update function in game.core.js
// -- data: packet send by server
function updateState (game, data){
  game.my_role = data.currStim.roles[game.my_id];
  game.condition = data.currStim.condition;
  game.playersThreshold = data.playersThreshold;
  game.currStim = _.shuffle(data.currStim.stimuli);
  game.active = data.active;
  game.roundNum = data.roundNum;
  game.roundStartTime = Date.now();
};

var customEvents = function(game) {
  $('.exitSurveyDropdown').change({game: game}, UI.dropdownTip);
  $('#surveySubmit').click({game: game}, UI.submit);
  
  // Tell server about clicking
  game.sendResponse = function(id) {
    const timeElapsed = Date.now() - game.messageSentTime;
    game.socket.send('clickedObj.' + id + '.' + timeElapsed);
  };

  // Tell server about typing
  $('#chatbox').on('input', function() {
    if($('#chatbox').val() != "" && !game.sentTyping) {
      game.socket.send('playerTyping.true');
      game.typingStartTime = Date.now();
      game.sentTyping = true;
    } else if($("#chatbox").val() == "") {
      game.socket.send('playerTyping.false');
      game.sentTyping = false;
    }
  });

  // Show typing when server tells us about it
  game.socket.on('playerTyping', function(data){
    if(data.typing == "true") {
      $('#messages')
	.append('<span class="typing-msg">Other player is typing...</span>')
	.stop(true,true)
	.animate({
	  scrollTop: $("#messages").prop("scrollHeight")
	}, 800);
    } else {
      $('.typing-msg').remove();
    }
  });

  // Send message when hitting enter...
  $('form').submit(function(){
    var origMsg = $('#chatbox').val();
    var timeElapsed = Date.now() - game.typingStartTime;
    var msg = ['chatMessage', origMsg.replace(/\./g, '~~~'), timeElapsed].join('.');
    if($('#chatbox').val() != '') {
      game.socket.send('playerTyping.false');
      game.socket.send(msg);
      game.sentTyping = false;
      UI.disableChatbox();
    }
    // This prevents the form from submitting & disconnecting person
    return false;
  });

  // Show model's message in box when we get it back from server
  game.socket.on('modelUtt', function(data) {
    game.socket.send(["chatMessage", data.replace(/\./g, '~~~'), 5000, 
		      'bot', 'speaker'].join('.'));
  });

  game.socket.on('modelResponse', function(data) {
    var id = _.find(game.currStim, {url: data})['targetStatus'];
    game.sendResponse(id);
  });

  game.socket.on('chatMessage', function(data){
    var source = game.my_role == game.playerRoleNames.role1 ? "you" : "speaker";
    var color = data.user === game.my_id ? "#363636" : "#707070";    
    // To bar responses until speaker has uttered at least one message
    game.messageSent = true;
    game.messageSentTime = Date.now();
    $('.typing-msg').remove();
    $('#messages')
      .append($('<li style="padding: 5px 10px; background: ' + color + '">')
    	      .text(source + ": " + data.msg))
      .stop(true,true)
      .animate({
	scrollTop: $("#messages").prop("scrollHeight")
      }, 800);
    if(game.bot.role == 'listener') {
      game.bot.respond(data.msg);
    }
  });

  game.socket.on('updateScore', function(data) {
    // update score
    var correct = data.outcome == 'target';
    var score = correct ? game.bonusAmt : 0;
    game.prevCorrect = correct;
    game.data.score += score;
    var bonus_score = (parseFloat(game.data.score) / 100
  		       .toFixed(2));
    // show feedback
    UI.highlightFeedback(game, data.outcome);
    $('#feedback').html(correct ? 'Nice! You got 2 points.' : 'Oh no! Your partner picked a different photo!');
    $('#score').empty().append('total points: ' + game.data.score);
    $('#messages').empty();
    $("#context").fadeOut(1000, function() {$(this).empty();});
    if(data.outcome == 'target') {
      UI.confetti.drop();
    }
  });
    
  game.socket.on('newRoundUpdate', function(data){
    if(data.active) {
      updateState(game, data);
      UI.reset(game, data);
    }
    // Kick things off by asking a question
    if(game.playersThreshold < 2) {
      game.bot = new Bot(game, data);
      if(game.bot.role == 'speaker') {
	game.bot.sampleUtterance();
      }
    }
  });
};

class Bot {
  constructor(game, data) {
    this.game = game;
    // opposite role of human participant
    this.role = (game.my_role == game.playerRoleNames.role1 ?
		 game.playerRoleNames.role2 : game.playerRoleNames.role1);
    console.log('initializing bot')
  }
  
  // Always asks about non-overlapping card
  sampleUtterance() {
    // remove revealed cards from goal set, so it won't keep asking about same card
    $('#messages')
      .append('<span class="typing-msg">Other player is typing...</span>')
      .stop(true,true)
      .animate({
	scrollTop: $("#messages").prop("scrollHeight")
      }, 800);
    
    // Request new model utterance
    const bundle = {
      action: 'speak',
      gameid: this.game.my_id, 
      roundNum : this.game.roundNum,
      target :  _.find(this.game.currStim, {targetStatus: 'target'})['url'],
      prevCorrect: this.game.prevCorrect == undefined ? false : this.game.prevCorrect,
      context: _.map(this.game.currStim, 'url')
    };
    
    console.log(bundle)
    this.game.socket.emit('requestModelAction', bundle);
  }
    // Always asks about non-overlapping card
  respond(caption) {
    // remove revealed cards from goal set, so it won't keep asking about same card
    setTimeout(function() {
      $('#messages')
	.append('<span class="typing-msg">Other player is thinking...</span>')
	.stop(true,true)
	.animate({
	  scrollTop: $("#messages").prop("scrollHeight")
	}, 800);
    }, 1000);

    const bundle = {
      action: 'listen',
      gameid: this.game.my_id, 
      roundNum : this.game.roundNum,
      target : _.find(this.game.currStim, {targetStatus: 'target'})['url'],
      caption: caption,
      feedback_img : _.find(this.game.currStim, {targetStatus: 'target'})['url'],
      context: _.map(this.game.currStim, 'url')
    };
    console.log(bundle)
    this.game.socket.emit('requestModelAction', bundle);
  }
}

module.exports = customEvents;
