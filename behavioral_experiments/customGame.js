const _ = require('lodash');
const fs    = require('fs');
const assert = require('assert');
const utils  = require(__base + 'src/sharedUtils.js');
const hard_contexts = require('../data/preprocess/coco_contexts_hard.json');
const easy_contexts = require('../data/preprocess/coco_contexts_easy.json');
const ServerGame = require('./src/game.js')['ServerGame'];
const sendPostRequest = require('request').post;

class ServerRefGame extends ServerGame {
  // eventually, will cycle through objects in context;
  // for now, we'll do what the model does, which is repeat 1 target k times
  constructor(config) {
    assert.ok(config.speakerType == 'human' || config.speakerType == 'bot');
    assert.ok(config.listenerType == 'human' || config.listenerType == 'bot');
    assert.ok(config.contextType == 'easy' || config.contextType == 'hard');
    const playersThreshold = (config.speakerType == 'human' &&
			      config.listenerType == 'human') ? 2 : 1;
    const contexts = config.contextType == 'easy' ? easy_contexts : hard_contexts;
    const context = _.clone(_.sample(contexts));
    
    super(_.extend({}, config, {playersThreshold}));
    this.playersThreshold = playersThreshold;
    this.context_id = context['cluster_ids'];
    this.context = context['neighbor_names'].slice(0, 4);
    this.numRepetitions = 6;
    this.humanRole = (config.speakerType == 'bot' ? 'listener' :
		      config.listenerType == 'bot' ? 'speaker' : 'both');
  }

  customEvents (socket) {
    // when bot asks us to generate an utterance,
    // we send a post request to port 5004 which tunnels 
    // to another machine with a GPU... 
    socket.on('requestModelAction', function(data) {
      console.log('sending request...')
      sendPostRequest('http://localhost:5005/request_model_action', {
	json: data
      }, (error, res, body) => {
	if (!error && res.statusCode === 200) {
	  _.forEach(socket.game.activePlayers(), p=> {
	    console.log('received response', body)
	    if(data.action == 'speak') {
	      p.player.instance.emit('modelUtt', body);
	    } else if(data.action == 'listen') {
	      p.player.instance.emit('modelResponse', body);
	    }
	  });
	} else {
	  console.log(`error getting stims: ${error} ${body}`);
	}
      });      
    });
  }
  
  // *
  // * TrialList creation
  // *
  
  makeTrialList () {
    // Keep sampling trial lists until we meet criterion
    let trialList = [];
    while(!this.checkTrialList(trialList)) {
      trialList = [];
      // Show each object once as target in each repetition block
      _.forEach(_.range(this.numRepetitions), repetition => {
	_.forEach(_.shuffle(this.context), target => {
	  trialList.push(this.sampleTrial(repetition, target));
	});
      });
    }
    return trialList;
  };

  checkTrialList (trialList) {
    const lengthMatch = trialList.length == 24 ;
    const noRepeats = _.every(utils.mapPairwise(trialList, function(curr, next) {
      return curr.targetImg.url !== next.targetImg.url ;
    }));
    return lengthMatch && noRepeats;
  }
  
  sampleTrial (repetition, targetUrl) {
    const target = {url: targetUrl , targetStatus : "target"};
    const distNums = _.range(this.context.length - 1);
    const distractors = _.map(_.without(this.context, targetUrl), d => {
      return {url: d, targetStatus: "distr" + distNums.pop()};
    });
    const roleNames = (this.playersThreshold == 1 ? [this.humanRole] : 
		     _.values(this.playerRoleNames));
    return {
      repNum: repetition,
      targetImg : target,
      stimuli: distractors.concat(target),
      roles: _.zipObject(_.map(this.players, p => p.id), roleNames)
    };
  };
  
  onMessage (client,message) {
    //Cut the message up into sub components
    const message_parts = message.split('.');

    //The first is always the type of message
    const message_type = message_parts[0];

    //Extract important variables
    const id = client.game.id;
    const all = client.game.activePlayers();
    const others = client.game.getOthers(client.userid);
    switch(message_type) {
      
    case 'chatMessage' :
      const paused = client.game.paused;
      const playersExist = client.game.playerCount == client.game.playersThreshold;
      if(playersExist && !paused) {
	const msg = message_parts[1].replace(/~~~/g,'.');
	_.map(all, p => p.player.instance.emit( 'chatMessage', {
	  user: client.userid, msg: msg
	}));
      }
      break;

    case 'playerTyping' :
      _.map(others, p => p.player.instance.emit( 'playerTyping', {
	typing: message_parts[1]
      }));
    break;

    case 'clickedObj' :
      _.map(all, p => p.player.instance.emit('updateScore', {
	outcome: message_parts[1]
      }));	
      client.game.newRound(2000);
      break; 

    case 'exitSurvey' :
      console.log(message_parts.slice(1));
      break;
      
    case 'h' : // Receive message when browser focus shifts
      //target.visible = message_parts[1];
      break;
    }
  };

  /*
    Associates events in onMessage with callback returning json to be saved
    {
    <eventName>: (client, message_parts) => {<datajson>}
    }
    Note: If no function provided for an event, no data will be written
  */
  dataOutput () {
    function commonOutput (client, message_data) {
      return {
    	iterationName: client.game.iterationName,
	context_id: client.game.context_id,
    	gameid: client.game.id,
    	time: Date.now(),
    	workerId: client.workerid,
    	assignmentId: client.assignmentid,
    	trialNum: client.game.roundNum,
	repNum: client.game.currStim.repNum,
	targetImg: client.game.currStim.targetImg.url
      };
    };
    
    function clickOutput (client, message_data) {
      const selection = message_data[1];
      const context = client.game.currStim.stimuli;
      return _.extend(
    	commonOutput(client, message_data), {
    	  timeFromMessage: message_data[2],
    	  clickedObj : selection,
	  correct : selection == 'target',
    	  fullContext: JSON.stringify(context)
    	});
    };
    
    function exitSurveyOutput (client, message_data) {
      const subjInfo = JSON.parse(message_data.slice(1));
      return _.extend(
    	_.omit(commonOutput(client, message_data),
    	       ['targetImg', 'repNum', 'trialNum', 'context_id']),
    	subjInfo);
    };
    

    function messageOutput (client, message_data) {
      const msg = message_data[1].replace(/~~~/g,'.');
      return _.extend(
    	commonOutput(client, message_data), {
	  msg: msg,
    	  timeFromRoundStart: message_data[2]
    	}
      );
    };

    return {
      'chatMessage' : messageOutput,
      'clickedObj' : clickOutput,
      'exitSurvey' : exitSurveyOutput
    };
  }
}

module.exports = ServerRefGame;
