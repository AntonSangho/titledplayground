// 물방울 생명체 게임 - 원 안에 격자 그리드와 여러 물방울 추가 (WEBGL 모드)

// 전역 변수 선언
let canvasSize; // 캔버스 크기
const GRID_SIZE = 25; // 25x25 그리드
let cellSize; // 각 셀의 크기 (픽셀)
let circleRadius; // 원의 반지름

// 기울기 관련 변수
let tiltX = 0; // X축 기울기 (좌우)
let tiltY = 0; // Y축 기울기 (상하)
let tiltZ = 0; // Z축 기울기 (회전)
const MAX_TILT = 30; // 최대 기울기 값 (도)
const TILT_STEP = 2; // 키 누를 때마다 기울어지는 정도 (도)

// 물방울 배열
let drops = [];
const NUM_DROPS = 10; // 물방울 개수
const DROP_RADIUS_MIN = 8; // 최소 물방울 반지름
const DROP_RADIUS_MAX = 15; // 최대 물방울 반지름
const DROP_SPEED = 2; // 초기 물방울 속도
const GRAVITY = 0.2; // 중력 가속도

// setup() 함수: 초기 설정
function setup() {
  // 캔버스 크기 설정
  canvasSize = min(windowWidth, windowHeight) * 0.8;
  
  // WEBGL 모드로 캔버스 생성
  createCanvas(canvasSize, canvasSize, WEBGL);
  
  // 각도 모드를 도(DEGREES)로 설정
  angleMode(DEGREES);
  
  // 원의 반지름 계산 (캔버스의 40%)
  circleRadius = canvasSize * 0.4;
  
  // 셀 크기 계산 (원 지름을 그리드 크기로 나눔)
  cellSize = (circleRadius * 2) / GRID_SIZE;
  
  // 물방울 생성
  createDrops();
}

// 물방울 생성 함수
function createDrops() {
  // 배열 초기화
  drops = [];
  
  // 여러 개의 물방울 생성
  for (let i = 0; i < NUM_DROPS; i++) {
    // 원 안에 랜덤 위치 생성
    let angle = random(360);
    let distance = random(circleRadius * 0.8); // 원 경계보다 안쪽에 생성
    let pos = createVector(
      cos(angle) * distance,
      sin(angle) * distance
    );
    
    // 랜덤 속도 생성
    let vel = p5.Vector.random2D();
    vel.mult(random(1, DROP_SPEED));
    
    // 랜덤 크기와 색상
    let radius = random(DROP_RADIUS_MIN, DROP_RADIUS_MAX);
    
    // HSB 색상 모델로 다양한 파란색 계열 생성
    colorMode(HSB, 360, 100, 100);
    let hue = random(180, 240); // 파란색 계열
    let saturation = random(70, 100);
    let brightness = random(70, 100);
    let dropColor = color(hue, saturation, brightness);
    colorMode(RGB, 255); // 다시 RGB 모드로 돌아감
    
    // 물방울 객체 생성 및 배열에 추가
    drops.push({
      position: pos,
      velocity: vel,
      radius: radius,
      color: dropColor
    });
  }
}

// draw() 함수: 반복 실행
function draw() {
  // 배경 다시 그리기
  background(240);
  
  // 조명 설정
  ambientLight(150);
  directionalLight(255, 255, 255, 0, 0, -1);
  
  // 원과 그리드 그리기
  push();
  // 키보드 입력에 따른 기울기 적용
  rotateX(tiltY); // Y값을 X축 회전에 적용 (위/아래 방향키)
  rotateY(tiltX); // X값을 Y축 회전에 적용 (좌/우 방향키)
  rotateZ(tiltZ); // Z값을 Z축 회전에 적용 (Z/X 키)
  
  drawCircleWithGrid();
  
  // 모든 물방울 업데이트 및 그리기
  for (let i = 0; i < drops.length; i++) {
    updateDrop(drops[i]);
    drawDrop(drops[i]);
  }
  
  pop();
}

// 원과 그리드를 함께 그리는 함수
function drawCircleWithGrid() {
  // 평면 위에 원과 그리드 그리기
  push();
  
  // 원 그리기 - 단색으로 표현
  fill(240); // 배경색과 같은 색
  stroke(100, 100, 255);
  strokeWeight(2);
  ellipse(0, 0, circleRadius * 2, circleRadius * 2);
  
  // 그리드 라인 그리기
  stroke(180);
  strokeWeight(1);
  
  // 수평선 그리기
  for (let j = -GRID_SIZE/2; j <= GRID_SIZE/2; j++) {
    let y = j * cellSize;
    // 원 안에 있는 선분만 그리기
    let maxX = sqrt(sq(circleRadius) - sq(y));
    line(-maxX, y, 0, maxX, y, 0);
  }
  
  // 수직선 그리기
  for (let i = -GRID_SIZE/2; i <= GRID_SIZE/2; i++) {
    let x = i * cellSize;
    // 원 안에 있는 선분만 그리기
    let maxY = sqrt(sq(circleRadius) - sq(x));
    line(x, -maxY, 0, x, maxY, 0);
  }
  
  // 중앙에 마름모 형태의 검정색 사각형 추가 (포탈)
  push();
  fill(0);
  noStroke();
  rotateZ(45); // 45도 회전 (마름모 형태로 만들기 위해)
  translate(0, 0, 1); // 약간 앞으로 이동시켜 그리드 위에 표시
  box(70, 70, 5); // 납작한 상자로 마름모 표현
  pop();
  
  pop();
}

// 물방울 업데이트 함수
function updateDrop(drop) {
  // 기울기에 따른 중력 가속도 벡터 생성
  let tiltXRad = radians(tiltX);
  let tiltYRad = radians(tiltY);
  
  // 기울기에 따른 중력 방향 계산
  let gravity = createVector(sin(tiltXRad), -sin(tiltYRad));
  gravity.mult(GRAVITY);
  
  // 중력 적용
  drop.velocity.add(gravity);
  
  // 위치 업데이트
  drop.position.add(drop.velocity);
  
  // 원 경계 충돌 처리
  handleDropCollision(drop);
  
  // 물방울 간 충돌 처리
  handleDropsCollision(drop);
}

// 원형 경계 충돌 처리 함수
function handleDropCollision(drop) {
  // 물방울의 중심과 원의 중심 사이의 거리
  let distance = drop.position.mag();
  
  // 충돌 감지: 물방울이 원의 경계에 접근했는지 확인
  if (distance + drop.radius > circleRadius) {
    // 법선 벡터 계산 (물방울 중심에서 원 중심 방향)
    let normal = drop.position.copy().normalize();
    
    // 속도 벡터를 법선 벡터에 대해 반사
    // v' = v - 2(v·n)n
    let dot = drop.velocity.dot(normal);
    let bounce = p5.Vector.mult(normal, 2 * dot);
    drop.velocity.sub(bounce);
    
    // 약간의 에너지 손실 (20%)
    drop.velocity.mult(0.8);
    
    // 물방울 위치 조정 (경계를 넘지 않도록)
    let overlap = distance + drop.radius - circleRadius;
    let correction = normal.copy().mult(overlap);
    drop.position.sub(correction);
  }
}

// 물방울 간 충돌 처리 함수
function handleDropsCollision(drop) {
  for (let i = 0; i < drops.length; i++) {
    let otherDrop = drops[i];
    
    // 자기 자신과의 충돌은 무시
    if (drop === otherDrop) continue;
    
    // 두 물방울 사이의 거리 계산
    let distance = p5.Vector.dist(drop.position, otherDrop.position);
    let minDist = drop.radius + otherDrop.radius;
    
    // 충돌 감지
    if (distance < minDist) {
      // 충돌 방향 벡터 계산
      let collisionDir = p5.Vector.sub(drop.position, otherDrop.position);
      collisionDir.normalize();
      
      // 겹침 정도 계산 및 위치 조정
      let overlap = minDist - distance;
      let correction = collisionDir.copy().mult(overlap * 0.5);
      drop.position.add(correction);
      otherDrop.position.sub(correction);
      
      // 속도 벡터 반사
      let relativeVel = p5.Vector.sub(drop.velocity, otherDrop.velocity);
      let speedTransfer = relativeVel.dot(collisionDir);
      
      // 양수일 때만 충돌 발생 (서로 가까워지는 경우)
      if (speedTransfer > 0) {
        // 질량 비율 (반지름의 비율로 계산)
        let m1 = drop.radius;
        let m2 = otherDrop.radius;
        let massRatio = m1 / (m1 + m2);
        
        // 충격량 계산
        let impulse = collisionDir.copy().mult(speedTransfer * 2.0 * massRatio);
        
        // 속도 업데이트
        drop.velocity.sub(impulse);
        otherDrop.velocity.add(impulse.mult(m1/m2));
        
        // 에너지 손실 (10%)
        drop.velocity.mult(0.9);
        otherDrop.velocity.mult(0.9);
      }
    }
  }
}

// 물방울 그리기 함수
function drawDrop(drop) {
  push();
  // 물방울 위치로 이동
  translate(drop.position.x, drop.position.y, drop.radius);
  
  // 물방울 색상 설정
  fill(drop.color);
  noStroke();
  
  // 물방울 그리기 (구체)
  sphere(drop.radius);
  
  pop();
}

// 키보드 키가 눌렸을 때 처리하는 함수
function keyPressed() {
  // 방향키로 기울기 조절
  if (keyCode === LEFT_ARROW) {
    // 왼쪽 키: 왼쪽이 눌리고 오른쪽이 올라옴
    tiltX = max(tiltX - TILT_STEP, -MAX_TILT);
  } else if (keyCode === RIGHT_ARROW) {
    // 오른쪽 키: 오른쪽이 눌리고 왼쪽이 올라옴
    tiltX = min(tiltX + TILT_STEP, MAX_TILT);
  } else if (keyCode === UP_ARROW) {
    // 위쪽 키: 위쪽이 눌리고 아래쪽이 올라옴
    tiltY = max(tiltY - TILT_STEP, -MAX_TILT);
  } else if (keyCode === DOWN_ARROW) {
    // 아래쪽 키: 아래쪽이 눌리고 위쪽이 올라옴
    tiltY = min(tiltY + TILT_STEP, MAX_TILT);
  } else if (key === 'z') {
    // Z키 - Z축 기울기 감소 (반시계 방향 회전)
    tiltZ = max(tiltZ - TILT_STEP, -MAX_TILT);
  } else if (key === 'x') {
    // X키 - Z축 기울기 증가 (시계 방향 회전)
    tiltZ = min(tiltZ + TILT_STEP, MAX_TILT);
  } else if (key === ' ') {
    // 스페이스바 - 기울기 초기화 및 물방울 리셋
    tiltX = 0;
    tiltY = 0;
    tiltZ = 0;
    createDrops(); // 모든 물방울 재생성
  } else if (key === 'a') {
    // a키 - 물방울 추가
    if (drops.length < 20) {
      // 원 안에 랜덤 위치 생성
      let angle = random(360);
      let distance = random(circleRadius * 0.7);
      let pos = createVector(
        cos(angle) * distance,
        sin(angle) * distance
      );
      
      // 랜덤 속도 생성
      let vel = p5.Vector.random2D();
      vel.mult(random(1, DROP_SPEED));
      
      // 랜덤 크기와 색상
      let radius = random(DROP_RADIUS_MIN, DROP_RADIUS_MAX);
      
      // HSB 색상 모델로 다양한 파란색 계열 생성
      colorMode(HSB, 360, 100, 100);
      let hue = random(180, 240); // 파란색 계열
      let saturation = random(70, 100);
      let brightness = random(70, 100);
      let dropColor = color(hue, saturation, brightness);
      colorMode(RGB, 255); // 다시 RGB 모드로 돌아감
      
      // 물방울 객체 생성 및 배열에 추가
      drops.push({
        position: pos,
        velocity: vel,
        radius: radius,
        color: dropColor
      });
    }
  }
  
  // 이벤트 기본 동작 방지 (페이지 스크롤 등)
  return false;
}

// 창 크기가 변경될 때 캔버스 크기도 조정
function windowResized() {
  canvasSize = min(windowWidth, windowHeight) * 0.8;
  resizeCanvas(canvasSize, canvasSize, WEBGL);
  
  // 원의 반지름 다시 계산
  circleRadius = canvasSize * 0.4;
  
  // 셀 크기 다시 계산
  cellSize = (circleRadius * 2) / GRID_SIZE;
}
