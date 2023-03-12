var j1 = require('./datasets/invasion_of_ukraine_by_russia.json')
var j2 = require('./datasets/russia_invasion.json')
var j3 = require('./datasets/russia_ukraine_conflict.json')
var j4 = require('./datasets/russia_ukraine_war.json')
var j5 = require('./datasets/ukraine_war.json')
const translate = require('@vitalets/google-translate-api');
var fs = require('fs')

async function mainModule() {
    var finalJsonSet = []
    var textMap = new Map()
    const jsons = [j1, j2, j3, j4, j5]
    jsons.forEach(async articleSet => {
        console.log(articleSet.claims.length)
        const newArticleSet = []
        for(let article of articleSet.claims){
            for(let retryCount = 0; retryCount < 5; retryCount++ ){
            if (!(article.claimReview[0].languageCode === 'en')) {
                if (article.claimReview[0].languageCode === 'zh') article.claimReview[0].languageCode = 'zh-CN'
                if (['fil', 'pa'].includes(article.claimReview[0].languageCode)) article.claimReview[0].languageCode = 'auto'
                try {
                    article.text = await translate(article.text, { from: article.claimReview[0].languageCode, to: 'en' })
                    article.claimReview[0].textualRating = await translate(article.claimReview[0].textualRating, { from: article.claimReview[0].languageCode, to: 'en' })
                    article.claimReview[0].languageCode = 'en'
                } catch (err) {
                    console.log(article)
                    console.log(err)
                    process.exit(1)
                }
            }}
            console.log(article.claimReview[0].languageCode)
            if(article.claimReview[0].languageCode != 'en'){
                throw new Error('Not Translated')
            }
            newArticleSet.push(article)
        }
        newArticleSet.claims.forEach(article => {
            if (!textMap.has(article.text)) {
                finalJsonSet.push(article)
                textMap[article.text] = true
            }
        })
    })
    console.log(finalJsonSet.length)
    fs.writeFileSync('./datasets/gfcmerged.json', JSON.stringify(finalJsonSet))
}

mainModule()