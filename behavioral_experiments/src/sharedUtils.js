var _ = require('lodash');
var fs = require('fs');
var path = require('path');
var mkdirp = require('mkdirp');
var sendPostRequest = require('request').post;

// Want to make sure there are no adjacent targets (e.g. gap is at least 1 apart?)
function mapPairwise(arr, func){
  var l = [];
  for(var i=0;i<arr.length-1;i++){
    l.push(func(arr[i], arr[i+1]));
  }
  return l;
}

function serveFile (req, res) {
  var fileName = req.params[0];
  return res.sendFile(fileName, {root: __base}); 
};

function handleDuplicate (req, res) {
  console.log("duplicate id: blocking request");
  return res.redirect('/src/duplicate.html');
};

function handleInvalidID (req, res) {
  console.log("invalid id: blocking request");
  return res.redirect('/src/invalid.html');
};

function checkPreviousParticipant (workerId, callback) {
  var p = {'workerId': workerId};
  var postData = {
    dbname: 'hri-conventions',
    query: p,
    projection: {'_id': 1}
  };
  sendPostRequest(
    'http://localhost:5002/db/exists',
    {json: postData},
    (error, res, body) => {
      try {
	if (!error && res.statusCode === 200) {
	  console.log("success! Received data " + JSON.stringify(body));
	  callback(body);
	} else {
	  throw `${error}`;
	}
      } catch (err) {
	console.log(err);
	console.log('no database; allowing participant to continue');
	return callback(false);
      }
    }
  );
};

function writeDataToCSV (game, _dataPoint) {
  var dataPoint = _.clone(_dataPoint);  
  var eventType = dataPoint.eventType;

  // Omit sensitive data
  if(game.anonymizeCSV) 
    dataPoint = _.omit(dataPoint, ['workerId', 'assignmentId']);
  
  // Establish stream to file if it doesn't already exist
  if(!_.has(game.streams, eventType))
    establishStream(game, dataPoint);    

  var line = _.values(dataPoint).join('\t') + "\n";
  game.streams[eventType].write(line, err => {if(err) throw err;});
};

function writeDataToMongo (game, line) {
  var postData = _.extend({
    dbname: game.projectName,
    colname: game.experimentName
  }, line);
  sendPostRequest(
    'http://localhost:5002/db/insert',
    { json: postData },
    (error, res, body) => {
      if (!error && res.statusCode === 200) {
        console.log(`sent data to store`);
      } else {
	console.log(`error sending data to store: ${error} ${body}`);
      }
    }
  );
};

function UUID() {
  var baseName = (Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10));
  var template = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx';
  var id = baseName + '-' + template.replace(/[xy]/g, function(c) {
    var r = Math.random()*16|0, v = c == 'x' ? r : (r&0x3|0x8);
    return v.toString(16);
  });
  return id;
};

function getLongFormTime () {
  var d = new Date();
  var day = [d.getFullYear(), (d.getMonth() + 1), d.getDate()].join('-');
  var time = [d.getHours() + 'h', d.getMinutes() + 'm', d.getSeconds() + 's'].join('-');
  return day + '-' + time;
};

function establishStream (game, dataPoint) {
  var startTime = getLongFormTime();
  var dirPath = ['.', 'data', dataPoint.eventType].join('/');
  var fileName = startTime + "-" + game.id + ".csv";
  var filePath = [dirPath, fileName].join('/');

  // Create path if it doesn't already exist
  mkdirp.sync(dirPath, err => {if (err) console.error(err);});

  // Write header
  var header = _.keys(dataPoint).join('\t') + '\n';
  fs.writeFile(filePath, header, err => {if(err) console.error(err);});

  // Create stream
  var stream = fs.createWriteStream(filePath, {'flags' : 'a'});
  game.streams[dataPoint.eventType] = stream;
};

function fillArray(value, len) {
  var arr = [];
  for (var i = 0; i < len; i++) {
    arr.push(value);
  }
  return arr;
}

module.exports = {
  mapPairwise,
  UUID,
  checkPreviousParticipant,
  serveFile,
  handleDuplicate,
  handleInvalidID,
  getLongFormTime,
  establishStream,
  writeDataToCSV,
  writeDataToMongo,
  fillArray
};
