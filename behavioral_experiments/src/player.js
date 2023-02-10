class GamePlayer {
  constructor (gameInstance, playerInstance) {
    this.instance = playerInstance;
    this.game = gameInstance;
    this.role = '';
    this.id = '';
  }
};

module.exports = GamePlayer;
