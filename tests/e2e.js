// End-to-end browser test: node tests/e2e.js  (needs: npm i -g playwright; server on :8021)
const {chromium}=require('playwright');
const BASE=process.env.BASE||'http://localhost:8021';
let fails=0;const ok=(c,m)=>{console.log((c?'PASS ':'FAIL ')+m);if(!c)fails++};
(async()=>{
const b=await chromium.launch();const p=await b.newPage({viewport:{width:1300,height:1000}});
const errs=[];p.on('pageerror',e=>errs.push(e.message));p.on('console',m=>{if(m.type()==='error')errs.push(m.text())});
const opts=()=>p.$$eval('#opts .opt',e=>e.length);

// 1. Random -> Approve shows options (the reported bug), with the progress bar appearing
await p.goto(BASE);
await p.click('#rand');await p.waitForSelector('#review:not([hidden])');
const barSeen=p.waitForFunction(()=>document.querySelector('#bar').classList.contains('on'),null,{timeout:5000}).then(()=>true,()=>false);
await p.click('#approve');await p.waitForSelector('#opts .opt',{timeout:30000});
ok(await opts()===3,'random: Approve shows 3 options');ok(await barSeen,'progress bar shown while working');
ok(await p.$$eval('#opts .playbtn',e=>e.every(x=>x.textContent.trim()==='Shoot')),'button is named Shoot');
ok(await p.$$eval('#opts .cut svg',e=>e.length)===3,'cut diagrams present');
// 2. group chooser re-ranks
await p.click('#seg2 button:nth-child(2)');await p.waitForTimeout(1500);ok(await opts()>=1,'group switch to solids keeps options');
// 3. Shoot x3 then Undo
for(let k=1;k<=3;k++){await p.click('#opts .opt:nth-child(1) .playbtn');await p.waitForFunction(n=>document.querySelector('#playnote').textContent.includes('Shot '+n),k,{timeout:30000});ok(await opts()>=1,'shoot #'+k+' re-ranks');}
await p.click('#undo');await p.waitForFunction(()=>document.querySelector('#playnote').textContent.includes('Undid'),null,{timeout:15000});ok(true,'undo works');
// 4. Back to review then Approve again
await p.click('#back');await p.waitForSelector('#review:not([hidden])');await p.click('#approve');await p.waitForSelector('#opts .opt');ok(await opts()>=1,'back + approve again works');
// 5. New random table after results
await p.click('#rand');await p.waitForSelector('#review:not([hidden])');await p.click('#approve');await p.waitForSelector('#opts .opt');ok(await opts()===3,'new random table after results -> approve works');
// 6. Tap edit: remove last point, add one, build, approve
await p.click('#back');await p.click('#tabTap');await p.waitForSelector('#ptable tbody tr');
const n0=await p.$$eval('#ptable tbody tr',e=>e.length);
await p.click('#ptable tbody tr:last-child button');await p.click('#photo',{position:{x:300,y:200}});
const v=await p.$$eval('#ptable tbody tr:last-child select option:not([disabled])',o=>o.map(x=>x.value).filter(Boolean)[0]);
await p.selectOption('#ptable tbody tr:last-child select',v);await p.click('#build');await p.waitForTimeout(1500);
ok(await p.$$eval('#all button',e=>e.length)===n0-6,'tap edit rebuilt table');
await p.click('#approve');await p.waitForSelector('#opts .opt');ok(await opts()>=1,'approve after edit');
// 7. Rails viewer
await p.click('#back');await p.click('#rails');await p.waitForSelector('#railStage svg');
ok(await p.$$eval('#railList button',e=>e.length)>=15,'rails: 15+ entries listed');
const t1=await p.textContent('#railTitle');await p.click('#railNext');ok((await p.textContent('#railTitle'))!==t1,'rails: Next changes the shot');
await p.click('#railFilter button:nth-child(3)');ok(await p.$$eval('#railList button',e=>e.length)===4,'rails: Banks filter = 4');
await p.click('#railsBack');ok(await p.isVisible('#review'),'rails: Back returns to review');
// 8. photo upload flow
await p.goto(BASE);await p.setInputFiles('#file',__dirname+'/fixtures/game_table.jpg');await p.click('#go');await p.waitForSelector('#review:not([hidden])',{timeout:30000});
await p.click('#approve');await p.waitForSelector('#opts .opt');ok(await opts()>=1,'photo upload -> approve');
// 9. press animation css present
ok(await p.evaluate(()=>[...document.styleSheets[0].cssRules].some(r=>r.selectorText&&r.selectorText.includes('button:active'))),'buttons have pressed state');
ok(errs.length===0,'no page errors '+JSON.stringify(errs));
await p.screenshot({path:'/tmp/e2e.png',fullPage:true});
await b.close();console.log(fails?`${fails} FAILED`:'ALL PASSED');process.exit(fails?1:0)})();
